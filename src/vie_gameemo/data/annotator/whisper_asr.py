"""ASR backends for Vietnamese game streaming audio.

Provides:
  WhisperASR      — faster-whisper backend (openai/whisper-* models)
  PhoWhisperASR   — HuggingFace transformers pipeline (vinai/PhoWhisper-*)
  BARTphoPostProcessor — optional seq2seq post-processing for text cleanup
  build_asr()     — factory that reads config and returns (asr, bartpho_or_None)
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


# ---------------------------------------------------------------------------
# PhoWhisper backend (VinAI — fine-tuned on Vietnamese)
# ---------------------------------------------------------------------------

class PhoWhisperASR:
    """ASR using VinAI PhoWhisper via HuggingFace transformers pipeline.

    PhoWhisper is Whisper fine-tuned on large-scale Vietnamese speech data.
    Significantly better than multilingual Whisper on pure Vietnamese audio,
    especially for Southern/Northern accents and gaming slang.

    Uses transformers.pipeline — no CTranslate2 conversion needed.
    """

    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-large",
        compute_type: str = "float16",
        chunk_length_s: int = 30,
        batch_size: int = 8,
        language: str = "vi",
    ) -> None:
        """Initialize PhoWhisperASR.

        Args:
            model_name: HF model ID (e.g. "vinai/PhoWhisper-large",
                "vinai/PhoWhisper-medium", "vinai/PhoWhisper-small").
            compute_type: "float16" on GPU, "float32" on CPU.
            chunk_length_s: Audio chunk size for long-form transcription.
                30s matches Whisper's native window.
            batch_size: Parallel chunks processed (higher = faster but more VRAM).
            language: Language code passed to generate_kwargs.
        """
        self.model_name = model_name
        self.compute_type = compute_type
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self.language = language
        self._pipe = None

    def load(self) -> None:
        """Load PhoWhisper via transformers ASR pipeline."""
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info("Loading PhoWhisper: %s (device=%s)", self.model_name, device)
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
        )
        logger.info("PhoWhisper loaded")

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to wav file.

        Returns:
            Transcribed text. Empty string on failure.
        """
        if self._pipe is None:
            raise RuntimeError("PhoWhisperASR not loaded. Call load() first.")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        result = self._pipe(
            str(audio_path),
            generate_kwargs={"language": self.language, "task": "transcribe"},
            return_timestamps=False,
        )
        text = result["text"].strip() if isinstance(result, dict) else ""
        logger.debug("PhoWhisper transcribed %s: %d chars", Path(audio_path).name, len(text))
        return text

    def batch_transcribe(self, audio_paths: list[Path]) -> list[str]:
        """Transcribe multiple files (pipeline handles batching internally).

        Args:
            audio_paths: List of wav paths.

        Returns:
            List of transcript strings in same order.
        """
        if self._pipe is None:
            raise RuntimeError("PhoWhisperASR not loaded. Call load() first.")
        results = self._pipe(
            [str(p) for p in audio_paths],
            generate_kwargs={"language": self.language, "task": "transcribe"},
            return_timestamps=False,
        )
        return [r["text"].strip() if isinstance(r, dict) else "" for r in results]


# ---------------------------------------------------------------------------
# BARTpho post-processor (optional text cleanup)
# ---------------------------------------------------------------------------

class BARTphoPostProcessor:
    """Optional seq2seq post-processing of ASR output using BARTpho.

    BARTpho (vinai/bartpho-syllable-1_5 or bartpho-word) is pre-trained
    with a denoising objective on Vietnamese text. Applied zero-shot with a
    correction prefix, it can clean up word-boundary errors, missing diacritics,
    and run-on words common in ASR output from noisy game audio.

    When to use:
      - Many merged words or missing spaces in transcript
      - Inconsistent diacritics (common with PhoWhisper on noisy audio)
      - Enabled via config: annotation.asr.bartpho.enabled = true

    Note: adds ~3 GB VRAM and ~5-10s per clip. Keep disabled unless quality
    warrants the cost.
    """

    def __init__(
        self,
        model_name: str = "vinai/bartpho-syllable-1_5",
        max_length: int = 256,
        num_beams: int = 4,
        prefix: str = "Sửa lỗi chính tả và hoàn thiện câu: ",
    ) -> None:
        """Initialize BARTphoPostProcessor.

        Args:
            model_name: HF model ID.
                "vinai/bartpho-syllable-1_5" — lighter, good for ASR cleanup.
                "vinai/bartpho-word" — heavier, better sentence structure.
            max_length: Max output token length.
            num_beams: Beam search width for generation.
            prefix: Instruction prepended to input text to guide denoising.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_beams = num_beams
        self.prefix = prefix
        self.tokenizer = None
        self.model = None
        self._device = None

    def load(self) -> None:
        """Load BARTpho tokenizer + model."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info("Loading BARTpho: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        ).to(self._device)
        self.model.eval()
        logger.info("BARTpho loaded")

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process(self, text: str) -> str:
        """Post-process a single transcript.

        Args:
            text: Raw ASR output.

        Returns:
            Cleaned text. Falls back to original if output is suspiciously
            shorter than input (generation went wrong).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("BARTphoPostProcessor not loaded. Call load() first.")
        if not text:
            return text

        import torch

        inp = self.prefix + text
        inputs = self.tokenizer(
            inp,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        cleaned = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # Safety guard: if output is much shorter than input, keep original
        if len(cleaned) < len(text) * 0.5:
            logger.debug("BARTpho output too short (%d vs %d chars); keeping original", len(cleaned), len(text))
            return text

        logger.debug("BARTpho: %d → %d chars", len(text), len(cleaned))
        return cleaned

    def batch_process(self, texts: list[str]) -> list[str]:
        """Post-process a list of transcripts sequentially.

        Args:
            texts: List of raw ASR strings.

        Returns:
            List of cleaned strings in same order.
        """
        return [self.process(t) for t in texts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_asr(asr_cfg) -> tuple:
    """Build ASR instance + optional BARTpho post-processor from config.

    Args:
        asr_cfg: SimpleNamespace with fields:
            backend: "whisper" | "phowhisper"
            whisper: sub-namespace with WhisperASR params
            phowhisper: sub-namespace with PhoWhisperASR params
            bartpho: sub-namespace with BARTphoPostProcessor params + .enabled

    Returns:
        (asr_instance, bartpho_instance_or_None)
        Both are NOT loaded — caller must call .load() before use.
    """
    backend = getattr(asr_cfg, "backend", "whisper")

    if backend == "phowhisper":
        ph = asr_cfg.phowhisper
        asr: WhisperASR | PhoWhisperASR = PhoWhisperASR(
            model_name=getattr(ph, "model_name", "vinai/PhoWhisper-large"),
            compute_type=getattr(ph, "compute_type", "float16"),
            chunk_length_s=getattr(ph, "chunk_length_s", 30),
            batch_size=getattr(ph, "batch_size", 8),
            language=getattr(ph, "language", "vi"),
        )
        logger.info("ASR backend: PhoWhisper (%s)", ph.model_name)
    else:
        w = getattr(asr_cfg, "whisper", asr_cfg)
        asr = WhisperASR(
            model_name=getattr(w, "model_name", "openai/whisper-large-v3"),
            compute_type=getattr(w, "compute_type", "int8_float16"),
            language=getattr(w, "language", "vi"),
            initial_prompt=getattr(w, "initial_prompt", _GAMING_INITIAL_PROMPT),
            vad_filter=getattr(w, "vad_filter", True),
            no_speech_threshold=getattr(w, "no_speech_threshold", 0.45),
            beam_size=getattr(w, "beam_size", 5),
            condition_on_previous_text=getattr(w, "condition_on_previous_text", False),
        )
        logger.info("ASR backend: Whisper (%s)", getattr(w, "model_name", "large-v3"))

    bartpho: BARTphoPostProcessor | None = None
    bt = getattr(asr_cfg, "bartpho", None)
    if bt is not None and getattr(bt, "enabled", False):
        bartpho = BARTphoPostProcessor(
            model_name=getattr(bt, "model_name", "vinai/bartpho-syllable-1_5"),
            max_length=getattr(bt, "max_length", 256),
            num_beams=getattr(bt, "num_beams", 4),
            prefix=getattr(bt, "prefix", "Sửa lỗi chính tả và hoàn thiện câu: "),
        )
        logger.info("BARTpho post-processing enabled (%s)", bt.model_name)

    return asr, bartpho
