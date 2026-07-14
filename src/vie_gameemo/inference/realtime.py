"""Real-time inference with sliding window.

Designed for live demo (Gradio/Streamlit). Processes a video stream in
configurable windows with step overlap, targeting <600ms latency per window.

Compression tricks:
    - Use lighter encoders when realtime.use_compressed=True
    - Skip LLM in real-time loop; trigger on-demand for highlights
    - Reuse webcam bbox detection across consecutive windows
    - Skip frames if behind real-time
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

_EMOTION_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]

# Bundled YOLOv11 webcam-region detector (fast; single class {0: 'webcam'}).
# Preferred over OWLv2 for the realtime demo — no 640MB download, ~instant.
_BUNDLED_WEBCAM_YOLO = Path(__file__).resolve().parent.parent / "utils" / "webcam_detect.pt"


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
        frames_per_window: int = 24,
        detect_webcam: bool = True,
        drop_slow_windows: bool = False,
        webcam_detector_path: Path | str | None = None,
        use_audio: bool = False,
        sample_fps: float = 4.0,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.cfg = cfg
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.max_latency_ms = max_latency_ms
        self.skip_llm = skip_llm
        # Frame sampling rate (Hz). Matches the validated batch pipeline (4 fps)
        # so per-window features are equivalent to the offline path.
        self.sample_fps = sample_fps
        # Cap on frames encoded per window (uniform subsample). Set high enough
        # to keep every sampled frame at sample_fps over the window.
        self.frames_per_window = frames_per_window
        # Detect the streamer webcam/facecam region once at stream start.
        self.detect_webcam = detect_webcam
        # Path to a YOLOv11 webcam detector (.pt). Defaults to the bundled one
        # if present; otherwise falls back to the config backend (OWLv2).
        if webcam_detector_path is not None:
            self.webcam_detector_path = Path(webcam_detector_path)
        elif _BUNDLED_WEBCAM_YOLO.exists():
            self.webcam_detector_path = _BUNDLED_WEBCAM_YOLO
        else:
            self.webcam_detector_path = None
        # When True, windows slower than max_latency_ms are dropped (live-stream
        # behaviour). For offline "analyse a long file" we keep every window.
        self.drop_slow_windows = drop_slow_windows

        self.cached_webcam_bbox = None
        self._webcam_cache_count = 0
        self._webcam_cache_max = 5

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fusion = None
        self._classifier = None
        # Encoders are instantiated once (lazily) and reused across all windows —
        # rebuilding a ViT per window would dominate runtime on long videos.
        self._face_enc = None
        self._ctx_enc = None
        self._audio_enc = None
        # Use real per-window audio (Whisper prosody) instead of zeros.
        self.use_audio = use_audio
        # Full-clip waveform, loaded once and sliced per window.
        self._audio_waveform = None
        self._audio_sr = 16000
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
        """Run on a video file/webcam, invoking on_prediction per window.

        Thin wrapper over :meth:`iter_predictions` for callback-style callers.
        """
        for result in self.iter_predictions(source):
            if on_prediction is not None:
                try:
                    on_prediction(result)
                except Exception as exc:
                    logger.warning("on_prediction callback error: %s", exc)

    def iter_predictions(self, source: Path | int):
        """Generator: yield one prediction dict per window as it is computed.

        Yielding (rather than a background thread + callback) lets a UI consume
        each batch the instant it finishes — control returns to the caller
        between windows so it can render before the next window is computed.

        Args:
            source: Video file path or camera device index.

        Yields:
            prediction_dict with keys: window_id, start_sec, end_sec, label,
            confidence, class_scores, has_face, webcam_found, latency_ms.
        """
        import cv2
        from collections import deque

        # Detect the webcam/facecam region once up-front so every window can
        # crop the streamer's face + context consistently (reused across windows).
        if self.detect_webcam and isinstance(source, Path):
            self._detect_webcam_once(source)

        # Load the full audio track once; per-window slices are cut in memory.
        if self.use_audio and isinstance(source, Path):
            self._load_audio_once(source)

        # Free GPU memory held by the one-shot YOLO detector before the loop.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step_frames = max(1, int(self.step_seconds * fps))
        # Sample at `sample_fps` (4 fps, matching the batch pipeline) and keep a
        # full window's worth of frames. Decimating native fps also bounds memory.
        target_fps = self.sample_fps
        keep_every = max(1, round(fps / target_fps))
        buf_maxlen = max(1, int(self.window_seconds * target_fps) + 2)
        # Rolling buffer holding at most the last `window_seconds` of (decimated) frames.
        buffer: deque = deque(maxlen=buf_maxlen)
        frame_idx = 0
        frames_since_emit = 0

        logger.info(
            "Processing stream (fps=%.1f, window=%ds, step=%ds, sample=%.1ffps)",
            fps, self.window_seconds, self.step_seconds, fps / keep_every,
        )

        def compute(start_sec: float, end_sec: float):
            t0 = time.perf_counter()
            result = self._process_window(list(buffer), start_sec, end_sec)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if self.drop_slow_windows and elapsed_ms > self.max_latency_ms:
                logger.debug(
                    "Window %d dropped (%.0fms > %.0fms limit)",
                    self._window_id, elapsed_ms, self.max_latency_ms,
                )
                return None
            result["latency_ms"] = round(elapsed_ms, 1)
            self._window_buffer.append(result)
            self._window_id += 1
            return result

        buffered_span = self.window_seconds  # seconds the full buffer spans
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                frames_since_emit += 1
                # Only keep every `keep_every`-th frame (decimation).
                if frame_idx % keep_every == 0:
                    buffer.append(frame)

                # Emit a prediction every `step` seconds, using the trailing
                # window. Starts from the first step (before a full window has
                # accumulated) so short clips still produce a timeline.
                if frames_since_emit >= step_frames and buffer:
                    frames_since_emit = 0
                    end_sec = frame_idx / fps
                    start_sec = max(0.0, end_sec - min(buffered_span, len(buffer) / target_fps))
                    result = compute(start_sec, end_sec)
                    if result is not None:
                        yield result

            # Flush the tail so the final partial step is not lost.
            if frames_since_emit > 0 and buffer:
                end_sec = frame_idx / fps
                start_sec = max(0.0, end_sec - min(buffered_span, len(buffer) / target_fps))
                result = compute(start_sec, end_sec)
                if result is not None:
                    yield result
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

        fusion_emb = window.get("fusion_emb")
        if fusion_emb is not None:
            evidence = {"fusion_emb": fusion_emb, "label": window.get("label", "neutral")}
        else:
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

    def _detect_webcam_once(self, source: Path) -> None:
        """Detect the streamer webcam/facecam bbox once and cache it.

        Prefers the bundled YOLOv11 detector (fast) when available, otherwise
        falls back to the config backend (OWLv2).
        """
        if self.cached_webcam_bbox is not None:
            return
        try:
            from vie_gameemo.preprocess.webcam_detector import WebcamDetector
            wc = getattr(getattr(self.cfg, "visual_encoder", None), "webcam_detector", None)
            if self.webcam_detector_path is not None:
                detector = WebcamDetector(
                    backend="yolo",
                    yolo_model=str(self.webcam_detector_path),
                    yolo_classes=[0],
                    min_detection_confidence=0.25,
                )
                logger.info("Using YOLO webcam detector: %s", self.webcam_detector_path)
            elif wc is not None:
                detector = WebcamDetector(
                    backend=getattr(wc, "backend", "owlv2"),
                    min_detection_confidence=getattr(wc, "min_detection_confidence", 0.1),
                    stability_threshold=getattr(wc, "stability_threshold", 0.3),
                )
            else:
                detector = WebcamDetector(backend="owlv2", min_detection_confidence=0.1)
            self.cached_webcam_bbox = detector.detect_webcam_region(source)
            logger.info("Webcam region detected: %s", self.cached_webcam_bbox is not None)
        except Exception as exc:
            logger.warning("Webcam detection failed (%s); using full-frame context", exc)
            self.cached_webcam_bbox = None

    def _load_audio_once(self, source: Path) -> None:
        """Extract + load the full audio waveform once (16kHz mono)."""
        if self._audio_waveform is not None:
            return
        try:
            import tempfile
            import librosa
            from vie_gameemo.preprocess.demux import extract_audio
            tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
            extract_audio(source, tmp_wav)
            self._audio_waveform, self._audio_sr = librosa.load(
                str(tmp_wav), sr=self._audio_sr, mono=True
            )
            tmp_wav.unlink(missing_ok=True)
            logger.info("Audio loaded: %.1fs", len(self._audio_waveform) / self._audio_sr)
        except Exception as exc:
            logger.warning("Audio load failed (%s); using zero audio", exc)
            self._audio_waveform = None

    def _process_window(
        self,
        frames: list,
        start_sec: float,
        end_sec: float,
    ) -> dict:
        """Extract features from a list of BGR frames and predict emotion."""
        # Uniformly subsample the window's frames to bound per-window cost.
        frames = _uniform_subsample(frames, self.frames_per_window)

        face_tensor, has_face = self._encode_faces(frames)
        ctx_tensor = self._encode_context(frames)
        audio_tensor = self._encode_audio_window(start_sec, end_sec)
        text_tensor = self._zero_text()

        fused = self._compute_fused(audio_tensor, face_tensor, ctx_tensor, text_tensor, has_face)
        prediction = self._predict_from_fused(fused)

        return {
            "window_id": self._window_id,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "class_scores": prediction["class_scores"],
            "has_face": bool(has_face),
            "webcam_found": self.cached_webcam_bbox is not None,
            "fusion_emb": fused.cpu(),  # stored for on-demand LLM reasoning
        }

    def _encode_faces(self, frames: list) -> tuple[torch.Tensor, bool]:
        """Encode streamer face crops from BGR frames → (1, T, 768), has_face."""
        if self.cached_webcam_bbox is None:
            # No facecam → no reliable face crop; use zeros placeholder.
            d = getattr(self.cfg.fusion, "d_model", 768)
            return torch.zeros(1, 1, d), False
        try:
            from vie_gameemo.preprocess.face_crop import extract_streamer_face
            if self._face_enc is None:
                from vie_gameemo.encoders.face_vit import FaceEncoder
                self._face_enc = FaceEncoder()
            crops = [extract_streamer_face(f, self.cached_webcam_bbox) for f in frames]
            tensor, has_face = self._face_enc.encode(crops)
            return tensor, has_face
        except Exception as exc:
            logger.debug("Face encoding failed: %s", exc)
            d = getattr(self.cfg.fusion, "d_model", 768)
            return torch.zeros(1, 1, d), False

    def _encode_context(self, frames: list) -> torch.Tensor:
        """Encode webcam-region (or full-frame) crops → (1, T, 768) context tensor."""
        try:
            if self._ctx_enc is None:
                from vie_gameemo.encoders.context_vit import ContextEncoder
                self._ctx_enc = ContextEncoder()
            enc = self._ctx_enc
            if self.cached_webcam_bbox is not None:
                from vie_gameemo.preprocess.face_crop import extract_webcam_region
                crops = [extract_webcam_region(f, self.cached_webcam_bbox) for f in frames]
                tensor = enc.encode(crops)
            else:
                tensor = enc.encode(list(frames)) if len(frames) else enc.encode(None)
            return tensor
        except Exception as exc:
            logger.debug("Context encoding failed: %s", exc)
            d = getattr(self.cfg.fusion, "d_model", 768)
            return torch.zeros(1, 1, d)

    def _zero_audio(self) -> torch.Tensor:
        """Return zero audio tensor (B=1, T=64, D=audio_dim)."""
        d = getattr(self.cfg.fusion, "audio_dim", None) or getattr(self.cfg.fusion, "d_model", 768)
        return torch.zeros(1, 64, d)

    def _encode_audio_window(self, start_sec: float, end_sec: float) -> torch.Tensor:
        """Encode the audio slice for [start_sec, end_sec] → (1, 64, 768).

        Falls back to zeros when audio is disabled/unavailable.
        """
        if not self.use_audio or self._audio_waveform is None:
            return self._zero_audio()
        try:
            sr = self._audio_sr
            a = int(max(0, start_sec) * sr)
            b = int(end_sec * sr)
            clip = self._audio_waveform[a:b]
            if clip.size == 0:
                return self._zero_audio()
            if self._audio_enc is None:
                from vie_gameemo.encoders.audio_whisper import WhisperAudioEncoder
                # Prefer GPU; fall back to CPU if VRAM is exhausted (device does
                # not change the result, only speed). Output is moved to the
                # fusion device later regardless.
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    self._audio_enc = WhisperAudioEncoder(device=dev)
                except RuntimeError as exc:
                    logger.warning("Audio encoder on %s failed (%s); using CPU", dev, exc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._audio_enc = WhisperAudioEncoder(device="cpu")
            return self._audio_enc.encode_waveform(clip)
        except Exception as exc:
            logger.debug("Audio encoding failed: %s", exc)
            return self._zero_audio()

    def _zero_text(self) -> torch.Tensor:
        """Return zero text tensor (B=1, T=1, D=text_dim).

        Text encoder output dim (e.g. 1024 for CafeBERT/XLM-R-large) may differ
        from the fused d_model; the per-modality MLP projects it down.
        """
        d = getattr(self.cfg.fusion, "text_dim", None) or getattr(self.cfg.fusion, "d_model", 768)
        return torch.zeros(1, 1, d)

    def _compute_fused(
        self,
        audio: torch.Tensor,
        face: torch.Tensor,
        context: torch.Tensor,
        text: torch.Tensor,
        has_face: bool,
    ) -> torch.Tensor:
        """Run encoders through fusion module, return (1, T, 768) fused tensor."""
        audio = audio.to(self._device)
        face = face.to(self._device)
        context = context.to(self._device)
        text = text.to(self._device)
        # Fusion expects has_face of shape (B,), not (B, 1).
        has_face_t = torch.tensor([has_face], dtype=torch.bool, device=self._device)

        with torch.no_grad():
            fused = self._fusion(audio, face, context, text, has_face=has_face_t)
            if isinstance(fused, tuple):
                fused = fused[0]
        return fused

    def _predict_from_fused(self, fused: torch.Tensor) -> dict:
        """Run classifier on fused tensor, return prediction dict."""
        with torch.no_grad():
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


def _uniform_subsample(frames: list, n: int) -> list:
    """Uniformly pick at most n frames from a window's frame list."""
    if n <= 0 or len(frames) <= n:
        return frames
    import numpy as np
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idx]
