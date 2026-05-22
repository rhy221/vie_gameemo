"""Whisper-large-v3 ASR via faster-whisper backend.

Transcribes audio (Vietnamese with English gaming slang code-switching).
Uses an initial prompt to bias toward gaming domain terms (clutch, ace, POG).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WhisperASR:
    """Whisper-large-v3 ASR wrapper using faster-whisper."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        compute_type: str = "float16",
        language: str = "vi",
        initial_prompt: str = "",
        vad_filter: bool = True,
        no_speech_threshold: float = 0.6,
    ) -> None:
        """Initialize Whisper.

        Args:
            model_name: HF model ID or faster-whisper size string.
            compute_type: "float16" | "int8_float16" | "int8".
            language: ISO 639-1 code (e.g., "vi").
            initial_prompt: Bias text (e.g., gaming slang vocabulary).
            vad_filter: Use VAD to skip silence (avoid hallucination).
            no_speech_threshold: If no-speech prob > threshold, output empty.
        """
        self.model_name = model_name
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter
        self.no_speech_threshold = no_speech_threshold
        self.model = None

    def load(self) -> None:
        """Load Whisper model via faster-whisper.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            ) from e

        # faster-whisper accepts HF model ID or size string like "large-v3"
        model_id = self.model_name
        if "whisper-" in model_id:
            # e.g., "openai/whisper-large-v3" → "large-v3"
            model_id = model_id.split("whisper-")[-1]

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = self.compute_type if device == "cuda" else "int8"

        logger.info("Loading Whisper model: %s (device=%s, compute=%s)", model_id, device, compute)
        self.model = WhisperModel(model_id, device=device, compute_type=compute)
        logger.info("Whisper model loaded")

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to wav file.

        Returns:
            Transcribed text. Empty string if all-silence or no-speech.

        Raises:
            RuntimeError: If model not loaded.
            FileNotFoundError: If audio_path missing.
        """
        if self.model is None:
            raise RuntimeError("WhisperASR not loaded. Call load() first.")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            initial_prompt=self.initial_prompt if self.initial_prompt else None,
            vad_filter=self.vad_filter,
            no_speech_threshold=self.no_speech_threshold,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("Transcribed %s: %d chars", audio_path.name, len(text))
        return text

    def batch_transcribe(self, audio_paths: list[Path]) -> list[str]:
        """Transcribe a batch sequentially (faster-whisper processes one at a time).

        Args:
            audio_paths: List of wav paths.

        Returns:
            List of transcript strings in same order.
        """
        return [self.transcribe(p) for p in audio_paths]
