"""Qwen2-Audio agent for audio prosody description (Catd).

Describes audio features (pitch, energy, speaking rate, presence of shouts/laughs)
in Vietnamese, used as one input to the consolidator.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _make_bnb_config(quantization: str):
    """Build BitsAndBytesConfig."""
    from transformers import BitsAndBytesConfig
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class QwenAudioAgent:
    """Audio prosody description agent using Qwen2-Audio."""

    def __init__(
        self,
        model_name: str,
        prompt: str,
        quantization: str = "4bit",
        max_new_tokens: int = 200,
    ) -> None:
        """Initialize Qwen2-Audio agent.

        Args:
            model_name: HF model ID (e.g., "Qwen/Qwen2-Audio-7B-Instruct").
            prompt: Vietnamese system prompt for prosody analysis.
            quantization: "4bit" | "8bit" | "none".
            max_new_tokens: Max tokens in description.
        """
        self.model_name = model_name
        self.prompt = prompt
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load Qwen2-Audio model + processor.

        Raises:
            ImportError: If transformers not installed or version too old.
        """
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2AudioForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "transformers with Qwen2-Audio support required. "
                "Run: pip install 'transformers>=4.45'"
            ) from e

        logger.info("Loading Qwen2-Audio: %s (quant=%s)", self.model_name, self.quantization)
        bnb_cfg = _make_bnb_config(self.quantization)

        kwargs: dict = {"device_map": "auto"}
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info("Qwen2-Audio loaded")

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded Qwen-Audio agent")

    def batch_describe(self, audio_paths: list[Path]) -> list[str]:
        """Generate Vietnamese prosody descriptions for a batch of audio files.

        Args:
            audio_paths: List of paths to wav files (16kHz mono).

        Returns:
            List of Vietnamese description strings.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("QwenAudioAgent not loaded. Call load() first.")

        import librosa
        import torch

        descriptions: list[str] = []
        for audio_path in audio_paths:
            try:
                audio, sr = librosa.load(str(audio_path), sr=self.processor.feature_extractor.sampling_rate, mono=True)

                conversation = [
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": str(audio_path)},
                        {"type": "text", "text": self.prompt},
                    ]}
                ]
                text_input = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                inputs = self.processor(
                    text=text_input,
                    audios=audio,
                    return_tensors="pt",
                    sampling_rate=sr,
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                    )

                out_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                descriptions.append(text)
                logger.debug("Audio described %s: %d chars", audio_path.name, len(text))

            except Exception as exc:
                logger.warning("Audio agent failed on %s: %s", audio_path, exc)
                descriptions.append("")

        return descriptions
