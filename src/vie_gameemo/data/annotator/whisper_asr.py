"""Whisper-large-v3 ASR via faster-whisper backend.

Transcribes audio (Vietnamese with English gaming slang code-switching).
Uses an initial prompt to bias toward gaming domain terms (clutch, ace, POG).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Gaming context prompt — biases Whisper toward Vietnamese gaming vocabulary
# and common code-switching patterns (Viet + English terms).
_GAMING_INITIAL_PROMPT = (
    "Đây là livestream game của streamer Việt Nam. "
    "Streamer hay nói: GG, clutch, ace, headshot, MVP, noob, lag, buff, nerf, "
    "rank, bot, carry, feed, gank, roam, farm, push, die, kill, team, "
    "ơi trời, vãi, thôi rồi, ăn rồi, xong rồi, đi nào, vào nào."
)


class WhisperASR:
    """Whisper-large-v3 ASR wrapper using faster-whisper."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        compute_type: str = "int8_float16",
        language: str = "vi",
        initial_prompt: str = _GAMING_INITIAL_PROMPT,
        vad_filter: bool = True,
        no_speech_threshold: float = 0.45,
        beam_size: int = 5,
        condition_on_previous_text: bool = False,
        log_prob_threshold: float = -1.0,
    ) -> None:
        """Initialize Whisper.

        Args:
            model_name: HF model ID or faster-whisper size string.
            compute_type: "float16" | "int8_float16" | "int8".
                int8_float16 is faster than float16 with near-identical quality.
            language: ISO 639-1 code (e.g., "vi").
            initial_prompt: Bias text (e.g., gaming slang vocabulary).
            vad_filter: Use VAD to skip silence (avoid hallucination).
            no_speech_threshold: Segments with no-speech prob above this are
                dropped. 0.45 is more permissive than default 0.6 — better for
                short 5s clips where Whisper is less confident.
            beam_size: Beam search width. 5 is the default.
            condition_on_previous_text: If True, previous segment text is used
                as context for next segment — can cause repetition loops in
                clips with game noise. Set False for short independent clips.
            log_prob_threshold: Segments below this avg log prob are dropped
                (hallucination filter). -1.0 means only very bad segments dropped.
        """
        self.model_name = model_name
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter
        self.no_speech_threshold = no_speech_threshold
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.log_prob_threshold = log_prob_threshold
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
            vad_parameters={"threshold": 0.4, "speech_pad_ms": 300},
            no_speech_threshold=self.no_speech_threshold,
            beam_size=self.beam_size,
            condition_on_previous_text=self.condition_on_previous_text,
            log_prob_threshold=self.log_prob_threshold,
            word_timestamps=False,
        )
        # Filter out segments flagged as likely hallucinations
        parts = []
        for seg in segments:
            if seg.no_speech_prob > self.no_speech_threshold:
                continue
            parts.append(seg.text.strip())
        text = " ".join(parts).strip()
        logger.debug(
            "Transcribed %s: %d chars (lang_prob=%.2f)",
            audio_path.name, len(text), info.language_probability,
        )
        return text

    def batch_transcribe(self, audio_paths: list[Path]) -> list[str]:
        """Transcribe a batch sequentially (faster-whisper processes one at a time).

        Args:
            audio_paths: List of wav paths.

        Returns:
            List of transcript strings in same order.
        """
        return [self.transcribe(p) for p in audio_paths]
