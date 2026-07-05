"""ASR backends for Vietnamese game streaming audio (bilingual VI+EN).

Provides:
  WhisperASR      — faster-whisper backend (openai/whisper-* models)
  PhoWhisperASR   — HuggingFace transformers pipeline (vinai/PhoWhisper-*)
  BARTphoPostProcessor — optional seq2seq post-processing for text cleanup
  FastTextLID     — fastText lid.176 language ID on transcript text
  build_asr()     — factory that reads config and returns (asr, bartpho_or_None)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_GAMING_PROMPT_VI = (
    "Đây là livestream game của streamer Việt Nam. "
    "Streamer hay nói: GG, clutch, ace, headshot, MVP, noob, lag, buff, nerf, "
    "rank, bot, carry, feed, gank, roam, farm, push, die, kill, team, "
    "ơi trời, vãi, thôi rồi, ăn rồi, xong rồi, đi nào, vào nào, "
    "địt mẹ, đéo, đụ má, cái lồn, vãi lồn, chó, ngu, đồ ngu, cứt, "
    "mẹ mày, bố mày, wtf, shit."
)

_GAMING_PROMPT_EN = (
    "This is a game livestream. The streamer often says: "
    "GG, clutch, ace, headshot, MVP, noob, lag, buff, nerf, "
    "rank, bot, carry, feed, gank, roam, farm, push, die, kill, team, "
    "let's go, oh my god, no way, come on, "
    "what the fuck, shit, damn, fuck, bro, dude."
)

_LANG_CONFIGS = {
    "vi": {"language": "vi", "initial_prompt": _GAMING_PROMPT_VI, "post_process": "bartpho"},
    "en": {"language": "en", "initial_prompt": _GAMING_PROMPT_EN, "post_process": "none"},
}


@dataclass
class TranscriptionResult:
    """Result from ASR transcription with language metadata."""
    text: str
    asr_detected_language: str | None = None
    asr_language_probability: float | None = None
    text_detected_language: str | None = None
    language_detect_confidence: float | None = None
    language_mismatch: bool = False


class FastTextLID:
    """Language identification using fastText lid.176 model."""

    def __init__(self, model_path: str = "lid.176.ftz") -> None:
        self.model_path = model_path
        self._model = None

    def load(self) -> None:
        try:
            import fasttext
        except ImportError as e:
            raise ImportError(
                "fasttext not installed. Run: pip install fasttext-wheel"
            ) from e

        resolved = Path(self.model_path)
        if not resolved.exists():
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            logger.info("Downloading lid.176.ftz from %s", url)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(resolved))

        logger.info("Loading fastText LID model: %s", resolved)
        self._model = fasttext.load_model(str(resolved))

    def predict(self, text: str) -> tuple[str, float]:
        """Predict language of text.

        Returns:
            (language_code, confidence) e.g. ("vi", 0.95).
        """
        if self._model is None:
            self.load()

        text_clean = text.replace("\n", " ").strip()
        if not text_clean:
            return ("unknown", 0.0)

        labels, probs = self._model.predict(text_clean, k=1)
        lang = labels[0].replace("__label__", "")
        return (lang, float(probs[0]))


class WhisperASR:
    """Whisper-large-v3 ASR wrapper using faster-whisper."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        compute_type: str = "int8_float16",
        vad_filter: bool = True,
        no_speech_threshold: float = 0.6,
        beam_size: int = 5,
        condition_on_previous_text: bool = False,
        log_prob_threshold: float = -0.5,
        hallucination_silence_threshold: float = 2.0,
        lang_configs: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.no_speech_threshold = no_speech_threshold
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.log_prob_threshold = log_prob_threshold
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.lang_configs = lang_configs or dict(_LANG_CONFIGS)
        self.model = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            ) from e

        model_id = self.model_name
        if "whisper-" in model_id:
            model_id = model_id.split("whisper-")[-1]

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = self.compute_type if device == "cuda" else "int8"

        logger.info("Loading Whisper model: %s (device=%s, compute=%s)", model_id, device, compute)
        self.model = WhisperModel(model_id, device=device, compute_type=compute)
        logger.info("Whisper model loaded")

    def unload(self) -> None:
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(
        self,
        audio_path: Path,
        source_language: str | None = None,
        routing: str = "metadata",
        lang_prob_threshold: float = 0.6,
    ) -> TranscriptionResult:
        """Transcribe a single audio file with language routing.

        Args:
            audio_path: Path to wav file.
            source_language: Ground-truth language from video metadata ("vi"/"en").
            routing: "metadata" | "auto" | "force".
            lang_prob_threshold: Below this, the initial pass's language pick
                is treated as unreliable (ambiguous/near-silent/noisy audio)
                even if it already landed inside the known candidate set —
                re-transcribing pinned to a concrete language + initial_prompt
                stabilizes decoding and reduces hallucination risk vs. letting
                Whisper decode under an uncertain language.

        Returns:
            TranscriptionResult with text and language metadata.
        """
        if self.model is None:
            raise RuntimeError("WhisperASR not loaded. Call load() first.")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        lang_cfg = self._resolve_lang_config(source_language, routing)
        language = lang_cfg.get("language")
        initial_prompt = lang_cfg.get("initial_prompt")

        transcribe_kwargs = {
            "language": language if routing != "auto" else None,
            "initial_prompt": initial_prompt,
            "vad_filter": self.vad_filter,
            "vad_parameters": {"threshold": 0.4, "speech_pad_ms": 300},
            "no_speech_threshold": self.no_speech_threshold,
            "beam_size": self.beam_size,
            "condition_on_previous_text": self.condition_on_previous_text,
            "log_prob_threshold": self.log_prob_threshold,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "word_timestamps": False,
        }

        segments, info = self.model.transcribe(str(audio_path), **transcribe_kwargs)

        if routing == "auto":
            # Whisper's open-set LID (~99 languages) is unreliable on short
            # (~5s) game-audio clips with background music/SFX, and can
            # confidently (>0.6) misdetect e.g. ko/zh even though this
            # dataset only ever contains the languages in `self.lang_configs`
            # (vi/en). Restrict the decision to those candidates using the
            # per-language probabilities Whisper already computed, instead of
            # trusting its unrestricted top-1 guess.
            candidates = tuple(self.lang_configs.keys())
            restricted_lang = info.language
            if info.all_language_probs:
                probs = dict(info.all_language_probs)
                restricted_lang = max(candidates, key=lambda l: probs.get(l, 0.0))
            elif restricted_lang not in candidates:
                restricted_lang = candidates[0]

            # Redo (pinned to a concrete language) if either: the open-set
            # top-1 landed outside the known candidates (wrong-language
            # case), OR confidence was low even though it stayed inside the
            # candidates (ambiguous/near-silent audio — forcing a language +
            # initial_prompt here stabilizes decoding and curbs hallucination,
            # same intent as the old flat confidence-threshold fallback).
            needs_redo = (
                restricted_lang != info.language
                or info.language_probability < lang_prob_threshold
            )
            if needs_redo:
                logger.info(
                    "Restricting LID for %s: open-set detected '%s' (p=%.2f) "
                    "-> forcing '%s' (dataset only contains %s)",
                    audio_path.name, info.language, info.language_probability,
                    restricted_lang, candidates,
                )
                lang_cfg = self.lang_configs.get(restricted_lang, _LANG_CONFIGS.get(restricted_lang, _LANG_CONFIGS["vi"]))
                transcribe_kwargs["language"] = restricted_lang
                transcribe_kwargs["initial_prompt"] = lang_cfg.get("initial_prompt")
                segments, info = self.model.transcribe(str(audio_path), **transcribe_kwargs)

        parts = []
        for seg in segments:
            if seg.no_speech_prob > self.no_speech_threshold:
                continue
            seg_text = seg.text.strip()
            if _is_hallucination(seg_text, seg):
                logger.debug("Filtered hallucinated segment: %r", seg_text[:80])
                continue
            parts.append(seg_text)
        text = " ".join(parts).strip()

        asr_detected = info.language
        mismatch = (
            source_language is not None
            and asr_detected is not None
            and asr_detected != source_language
        )

        logger.debug(
            "Transcribed %s: %d chars (detected=%s, prob=%.2f, source=%s, mismatch=%s)",
            audio_path.name, len(text), asr_detected,
            info.language_probability, source_language, mismatch,
        )

        return TranscriptionResult(
            text=text,
            asr_detected_language=asr_detected,
            asr_language_probability=info.language_probability,
            language_mismatch=mismatch,
        )

    def batch_transcribe(
        self,
        audio_paths: list[Path],
        source_language: str | None = None,
        routing: str = "metadata",
    ) -> list[TranscriptionResult]:
        return [self.transcribe(p, source_language=source_language, routing=routing) for p in audio_paths]

    def _resolve_lang_config(self, source_language: str | None, routing: str) -> dict:
        if routing == "metadata":
            lang = source_language or "vi"
            return self.lang_configs.get(lang, _LANG_CONFIGS.get(lang, _LANG_CONFIGS["vi"]))
        elif routing == "force":
            return self.lang_configs.get("vi", _LANG_CONFIGS["vi"])
        else:  # auto
            return self.lang_configs.get(source_language or "vi", _LANG_CONFIGS["vi"])


class PhoWhisperASR:
    """ASR using VinAI PhoWhisper via HuggingFace transformers pipeline.

    PhoWhisper is fine-tuned on Vietnamese — does NOT support English.
    If source_language=="en", logs a warning and returns None to signal
    the caller to fallback to WhisperASR.
    """

    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-large",
        compute_type: str = "float16",
        chunk_length_s: int = 30,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self._pipe = None

    def load(self) -> None:
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
        import gc
        import torch
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(
        self,
        audio_path: Path,
        source_language: str | None = None,
    ) -> TranscriptionResult | None:
        """Transcribe a single audio file.

        Returns None if source_language is "en" (PhoWhisper doesn't support EN).
        Caller should fallback to WhisperASR.
        """
        if source_language == "en":
            logger.warning(
                "PhoWhisper does not support English; returning None for fallback. "
                "Clip: %s", audio_path
            )
            return None

        if self._pipe is None:
            raise RuntimeError("PhoWhisperASR not loaded. Call load() first.")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        result = self._pipe(
            str(audio_path),
            generate_kwargs={"language": "vi", "task": "transcribe"},
            return_timestamps=False,
        )
        text = result["text"].strip() if isinstance(result, dict) else ""
        logger.debug("PhoWhisper transcribed %s: %d chars", Path(audio_path).name, len(text))

        return TranscriptionResult(
            text=text,
            asr_detected_language="vi",
        )

    def batch_transcribe(
        self,
        audio_paths: list[Path],
        source_language: str | None = None,
    ) -> list[TranscriptionResult | None]:
        return [self.transcribe(p, source_language=source_language) for p in audio_paths]


class BARTphoPostProcessor:
    """Optional seq2seq post-processing of ASR output using BARTpho.

    Only applies to Vietnamese text. English transcripts skip this step.
    """

    def __init__(
        self,
        model_name: str = "vinai/bartpho-syllable-1_5",
        max_length: int = 256,
        num_beams: int = 4,
        prefix: str = "Sửa lỗi chính tả và hoàn thiện câu: ",
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.num_beams = num_beams
        self.prefix = prefix
        self.tokenizer = None
        self.model = None
        self._device = None

    def load(self) -> None:
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
        import gc
        import torch
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process(self, text: str) -> str:
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

        if len(cleaned) < len(text) * 0.5:
            logger.debug("BARTpho output too short (%d vs %d chars); keeping original", len(cleaned), len(text))
            return text

        logger.debug("BARTpho: %d → %d chars", len(text), len(cleaned))
        return cleaned

    def batch_process(self, texts: list[str]) -> list[str]:
        return [self.process(t) for t in texts]


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

import re

_HALLUCINATION_PATTERNS = [
    re.compile(r"subscribe", re.IGNORECASE),
    re.compile(r"đăng\s*ký", re.IGNORECASE),
    re.compile(r"like\s*(và|and)\s*share", re.IGNORECASE),
    re.compile(r"kênh\s+\w+", re.IGNORECASE),
    re.compile(r"cảm\s*ơn.*theo\s*dõi", re.IGNORECASE),
    re.compile(r"đừng\s*quên", re.IGNORECASE),
    re.compile(r"nhấn\s*(nút|chuông)", re.IGNORECASE),
    re.compile(r"bỏ\s*lỡ", re.IGNORECASE),
    re.compile(r"video\s*(tiếp|sau|mới)", re.IGNORECASE),
    re.compile(r"(phụ đề|subtitle).*tự động", re.IGNORECASE),
    re.compile(r"www\.|\.com|\.vn|http", re.IGNORECASE),
]


def _is_hallucination(text: str, segment=None) -> bool:
    """Check if a transcribed segment is likely a Whisper hallucination."""
    if not text:
        return False

    for pattern in _HALLUCINATION_PATTERNS:
        if pattern.search(text):
            return True

    # Repetitive text (same phrase looped) — classic hallucination sign
    words = text.split()
    if len(words) >= 6:
        half = len(words) // 2
        if words[:half] == words[half:2 * half]:
            return True

    # Segment-level heuristic: very low avg_logprob often means hallucination
    if segment is not None:
        avg_logprob = getattr(segment, "avg_logprob", 0.0)
        if avg_logprob < -0.7:
            return True

    return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_asr(asr_cfg) -> tuple:
    """Build ASR instance + optional BARTpho post-processor from config.

    Args:
        asr_cfg: SimpleNamespace with fields:
            backend: "whisper" | "phowhisper"
            language_routing: "metadata" | "auto" | "force"
            whisper: sub-namespace with per-language configs
            phowhisper: sub-namespace with PhoWhisperASR params
            bartpho: sub-namespace with BARTphoPostProcessor params + .enabled
            text_lid: sub-namespace with fastText LID config

    Returns:
        (asr_instance, bartpho_instance_or_None)
        Both are NOT loaded — caller must call .load() before use.
    """
    backend = getattr(asr_cfg, "backend", "whisper")

    whisper_fallback: WhisperASR | None = None

    if backend == "phowhisper":
        ph = asr_cfg.phowhisper
        asr: WhisperASR | PhoWhisperASR = PhoWhisperASR(
            model_name=getattr(ph, "model_name", "vinai/PhoWhisper-large"),
            compute_type=getattr(ph, "compute_type", "float16"),
            chunk_length_s=getattr(ph, "chunk_length_s", 30),
            batch_size=getattr(ph, "batch_size", 8),
        )
        w = getattr(asr_cfg, "whisper", None)
        if w is not None:
            lang_configs = _parse_lang_configs(w)
            whisper_fallback = WhisperASR(
                model_name=getattr(w, "model_name", "openai/whisper-large-v3"),
                compute_type=getattr(w, "compute_type", "int8_float16"),
                vad_filter=getattr(w, "vad_filter", True),
                no_speech_threshold=getattr(w, "no_speech_threshold", 0.6),
                beam_size=getattr(w, "beam_size", 5),
                condition_on_previous_text=getattr(w, "condition_on_previous_text", False),
                log_prob_threshold=getattr(w, "log_prob_threshold", -0.5),
                hallucination_silence_threshold=getattr(w, "hallucination_silence_threshold", 2.0),
                lang_configs=lang_configs,
            )
        logger.info("ASR backend: PhoWhisper (%s) with Whisper fallback for EN", ph.model_name)
    else:
        w = getattr(asr_cfg, "whisper", asr_cfg)
        lang_configs = _parse_lang_configs(w)
        asr = WhisperASR(
            model_name=getattr(w, "model_name", "openai/whisper-large-v3"),
            compute_type=getattr(w, "compute_type", "int8_float16"),
            vad_filter=getattr(w, "vad_filter", True),
            no_speech_threshold=getattr(w, "no_speech_threshold", 0.6),
            beam_size=getattr(w, "beam_size", 5),
            condition_on_previous_text=getattr(w, "condition_on_previous_text", False),
            log_prob_threshold=getattr(w, "log_prob_threshold", -0.5),
            hallucination_silence_threshold=getattr(w, "hallucination_silence_threshold", 2.0),
            lang_configs=lang_configs,
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

    # Attach extra attributes for the pipeline to use
    asr._whisper_fallback = whisper_fallback  # type: ignore[attr-defined]
    asr._asr_cfg = asr_cfg  # type: ignore[attr-defined]

    return asr, bartpho


_lid_cache: dict[str, FastTextLID] = {}


def _get_lid(model_path: str) -> FastTextLID:
    if model_path not in _lid_cache:
        _lid_cache[model_path] = FastTextLID(model_path=model_path)
    return _lid_cache[model_path]


def transcribe_clip(
    asr: WhisperASR | PhoWhisperASR,
    bartpho: BARTphoPostProcessor | None,
    audio_path: Path,
    source_language: str = "vi",
    asr_cfg=None,
) -> TranscriptionResult:
    """High-level transcription with routing, fallback, post-processing, and LID cross-check.

    This is the main entry point for the annotation pipeline.
    """
    if asr_cfg is None:
        asr_cfg = getattr(asr, "_asr_cfg", None)

    routing = "metadata"
    lang_prob_threshold = 0.6
    if asr_cfg is not None:
        routing = getattr(asr_cfg, "language_routing", "metadata")
        lang_prob_threshold = getattr(asr_cfg, "lang_prob_threshold", 0.6)

    if routing == "force":
        force_lang = getattr(asr_cfg, "force_language", "vi") if asr_cfg else "vi"
        source_language = force_lang

    # Transcribe
    if isinstance(asr, PhoWhisperASR):
        result = asr.transcribe(audio_path, source_language=source_language)
        if result is None:
            # PhoWhisper can't do EN — fallback to Whisper
            fallback = getattr(asr, "_whisper_fallback", None)
            if fallback is None:
                raise RuntimeError(
                    f"PhoWhisper cannot transcribe EN and no Whisper fallback configured "
                    f"for {audio_path}"
                )
            if fallback.model is None:
                fallback.load()
            result = fallback.transcribe(
                audio_path,
                source_language=source_language,
                routing=routing,
                lang_prob_threshold=lang_prob_threshold,
            )
    else:
        result = asr.transcribe(
            audio_path,
            source_language=source_language,
            routing=routing,
            lang_prob_threshold=lang_prob_threshold,
        )

    # Post-process: BARTpho only for VI
    lang_cfg = _LANG_CONFIGS.get(source_language, {})
    post_process = lang_cfg.get("post_process", "none")
    if bartpho is not None and post_process == "bartpho" and result.text:
        result.text = bartpho.process(result.text)

    # fastText LID cross-check
    detect_for_validation = True
    if asr_cfg is not None:
        detect_for_validation = getattr(asr_cfg, "detect_for_validation", True)

    if detect_for_validation and result.text:
        try:
            text_lid_cfg = getattr(asr_cfg, "text_lid", None) if asr_cfg else None
            model_path = getattr(text_lid_cfg, "model", "lid.176.ftz") if text_lid_cfg else "lid.176.ftz"
            lid = _get_lid(model_path)
            lang_code, confidence = lid.predict(result.text)
            result.text_detected_language = lang_code
            result.language_detect_confidence = confidence
            if lang_code != source_language:
                result.language_mismatch = True
                logger.info(
                    "LID mismatch for %s: source=%s, text_detected=%s (conf=%.2f)",
                    audio_path.name, source_language, lang_code, confidence,
                )
        except Exception as exc:
            logger.warning("fastText LID failed for %s: %s", audio_path.name, exc)

    return result


def _parse_lang_configs(whisper_ns) -> dict:
    """Extract per-language configs from whisper config namespace."""
    configs = {}
    for lang in ("vi", "en"):
        lang_ns = getattr(whisper_ns, lang, None)
        if lang_ns is not None:
            configs[lang] = {
                "language": getattr(lang_ns, "language", lang),
                "initial_prompt": getattr(lang_ns, "initial_prompt", _LANG_CONFIGS.get(lang, {}).get("initial_prompt", "")),
                "post_process": getattr(lang_ns, "post_process", "none"),
            }
        else:
            configs[lang] = _LANG_CONFIGS.get(lang, {"language": lang})
    return configs
