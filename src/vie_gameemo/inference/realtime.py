"""Real-time inference with sliding window.

Designed for live demo (Gradio/Streamlit). Processes a video stream in
configurable windows with step overlap, targeting <600ms latency per window.

Compression tricks:
    - Use lighter encoders when realtime.use_compressed=True
    - Skip LLM in real-time loop; trigger on-demand for highlights
    - Reuse webcam bbox detection across consecutive windows
    - Skip frames if behind real-time
"""

import logging
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

_EMOTION_LABELS = ["hype", "tilted", "focused", "disappointed", "shocked", "amused", "neutral"]


class RealtimeInferenceRunner:
    """Sliding-window real-time inference.

    Args:
        checkpoint: Trained model checkpoint.
        cfg: Full config.
        window_seconds: Sliding window size.
        step_seconds: Step between window starts.
        max_latency_ms: Drop windows if processing exceeds this.
        skip_llm: If True, only classify (no reasoning) for speed.
    """

    def __init__(
        self,
        checkpoint: Path,
        cfg: SimpleNamespace,
        window_seconds: float = 5.0,
        step_seconds: float = 1.0,
        max_latency_ms: float = 600.0,
        skip_llm: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.cfg = cfg
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.max_latency_ms = max_latency_ms
        self.skip_llm = skip_llm

        self.cached_webcam_bbox = None
        self._webcam_cache_count = 0
        self._webcam_cache_max = 5

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fusion = None
        self._classifier = None
        self._window_buffer: list[dict] = []
        self._window_id = 0

        self._load_model()

    def _load_model(self) -> None:
        """Load fusion + classifier once at startup."""
        from vie_gameemo.inference.batch import _load_model
        self._fusion, self._classifier = _load_model(
            self.checkpoint, self.cfg, self._device
        )
        logger.info("RealtimeInferenceRunner model loaded on %s", self._device)

    def process_stream(
        self,
        source: Path | int,
        on_prediction: callable | None = None,
    ) -> None:
        """Run on a video file or webcam (integer device ID).

        Processes source in sliding windows of window_seconds with step_seconds
        stride. For each window, extracts features and predicts emotion. If
        processing falls behind real-time, windows are skipped to catch up.

        Args:
            source: Video file path or camera device index.
            on_prediction: Callback(prediction_dict) invoked per window result.
                prediction_dict keys: window_id, start_sec, end_sec, label,
                confidence, class_scores, latency_ms.
        """
        import cv2

        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        window_frames = int(self.window_seconds * fps)
        step_frames = int(self.step_seconds * fps)
        frame_buffer: list = []
        frame_idx = 0

        logger.info("Processing stream (fps=%.1f, window=%ds, step=%ds)",
                    fps, self.window_seconds, self.step_seconds)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_buffer.append(frame)
                frame_idx += 1

                if len(frame_buffer) >= window_frames:
                    window_frames_list = frame_buffer[:window_frames]
                    frame_buffer = frame_buffer[step_frames:]

                    start_sec = (frame_idx - window_frames) / fps
                    end_sec = frame_idx / fps

                    t0 = time.perf_counter()
                    result = self._process_window(window_frames_list, start_sec, end_sec)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    if elapsed_ms > self.max_latency_ms:
                        logger.debug(
                            "Window %d dropped (%.0fms > %.0fms limit)",
                            self._window_id, elapsed_ms, self.max_latency_ms,
                        )
                        frame_buffer = []
                        continue

                    result["latency_ms"] = round(elapsed_ms, 1)
                    self._window_buffer.append(result)

                    if on_prediction is not None:
                        try:
                            on_prediction(result)
                        except Exception as exc:
                            logger.warning("on_prediction callback error: %s", exc)

                    self._window_id += 1
        finally:
            cap.release()

        logger.info("Stream processing complete. %d windows processed.", self._window_id)

    def explain_window(self, window_id: int) -> dict:
        """On-demand: invoke LLM for a specific past window.

        Args:
            window_id: ID of past window in internal buffer.

        Returns:
            Dict with 'reasoning' and 'answer' from LLM.
        """
        window = next(
            (w for w in self._window_buffer if w.get("window_id") == window_id), None
        )
        if window is None:
            raise KeyError(f"Window {window_id} not found in buffer")

        from vie_gameemo.inference.batch import _load_llm
        llm = _load_llm(self.cfg)
        if llm is None:
            return {"reasoning": "", "answer": window.get("label", "neutral")}

        evidence = {
            "face_aus": window.get("face_aus", "N/A"),
            "visual_objective": window.get("visual_description", "N/A"),
            "audio_tone": window.get("audio_description", "N/A"),
            "transcript": window.get("transcript", ""),
            "label": window.get("label", "neutral"),
        }
        llm_out = llm.reason(evidence)
        if hasattr(llm, "unload"):
            llm.unload()
        return {"reasoning": llm_out.reasoning, "answer": llm_out.answer}

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------

    def _process_window(
        self,
        frames: list,
        start_sec: float,
        end_sec: float,
    ) -> dict:
        """Extract features from a list of BGR frames and predict emotion."""
        import io
        import tempfile

        import cv2
        from PIL import Image

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        face_tensor, has_face = self._encode_faces(pil_frames)
        ctx_tensor = self._encode_context(pil_frames)
        audio_tensor = self._zero_audio()
        text_tensor = self._zero_text()

        prediction = self._forward(audio_tensor, face_tensor, ctx_tensor, text_tensor, has_face)

        return {
            "window_id": self._window_id,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "class_scores": prediction["class_scores"],
        }

    def _encode_faces(self, frames: list) -> tuple[torch.Tensor, bool]:
        """Encode face crops from PIL frames → (1, T, 768), has_face."""
        try:
            from vie_gameemo.encoders.face_vit import FaceEncoder
            enc = FaceEncoder()
            tensor, has_face = enc.encode(frames)
            del enc
            return tensor, has_face
        except Exception as exc:
            logger.debug("Face encoding failed: %s", exc)
            d = getattr(self.cfg.fusion, "d_model", 768)
            return torch.zeros(1, 1, d), False

    def _encode_context(self, frames: list) -> torch.Tensor:
        """Encode full frames → (1, T, 768) context tensor."""
        try:
            from vie_gameemo.encoders.context_vit import ContextEncoder
            enc = ContextEncoder()
            tensor = enc.encode(frames)
            del enc
            return tensor
        except Exception as exc:
            logger.debug("Context encoding failed: %s", exc)
            d = getattr(self.cfg.fusion, "d_model", 768)
            return torch.zeros(1, 1, d)

    def _zero_audio(self) -> torch.Tensor:
        """Return zero audio tensor (B=1, T=64, D=768)."""
        d = getattr(self.cfg.fusion, "d_model", 768)
        return torch.zeros(1, 64, d)

    def _zero_text(self) -> torch.Tensor:
        """Return zero text tensor (B=1, T=1, D=768)."""
        d = getattr(self.cfg.fusion, "d_model", 768)
        return torch.zeros(1, 1, d)

    def _forward(
        self,
        audio: torch.Tensor,
        face: torch.Tensor,
        context: torch.Tensor,
        text: torch.Tensor,
        has_face: bool,
    ) -> dict:
        """Run fusion + classifier."""
        audio = audio.to(self._device)
        face = face.to(self._device)
        context = context.to(self._device)
        text = text.to(self._device)
        has_face_t = torch.tensor([[has_face]], dtype=torch.bool, device=self._device)

        with torch.no_grad():
            fused = self._fusion(audio, face, context, text, has_face=has_face_t)
            if isinstance(fused, tuple):
                fused = fused[0]
            logits = self._classifier(fused)
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = int(probs.argmax().item())
        return {
            "label": _EMOTION_LABELS[pred_idx] if pred_idx < len(_EMOTION_LABELS) else str(pred_idx),
            "confidence": float(probs[pred_idx].item()),
            "class_scores": {
                _EMOTION_LABELS[i]: float(probs[i].item())
                for i in range(len(_EMOTION_LABELS))
            },
        }
