"""Qwen2.5-VL agent for visual objective description (Cvod).

Describes the scene/context in a single frame using Qwen2.5-VL.
Output is plain Vietnamese text describing setting, posture, environment —
NOT emotion (that comes from the consolidator combining all signals).

Memory: model is loaded once per batch, unloaded after. Use 4-bit quant.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _make_bnb_config(quantization: str):
    """Build BitsAndBytesConfig for 4-bit or 8-bit quantization."""
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


class QwenVLAgent:
    """Visual scene description agent using Qwen2.5-VL."""

    def __init__(
        self,
        model_name: str,
        prompt: str,
        quantization: str = "4bit",
        max_new_tokens: int = 200,
        temperature: float = 0.3,
    ) -> None:
        """Initialize Qwen2.5-VL agent.

        Args:
            model_name: HF model ID (e.g., "Qwen/Qwen2.5-VL-7B-Instruct").
            prompt: System prompt template (Vietnamese).
            quantization: "4bit" | "8bit" | "none".
            max_new_tokens: Max tokens in description.
            temperature: Sampling temperature.
        """
        self.model_name = model_name
        self.prompt = prompt
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load Qwen2.5-VL model and processor into GPU memory.

        Raises:
            ImportError: If transformers or bitsandbytes is not installed.
        """
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "transformers>=4.45 with Qwen2.5-VL support required. "
                "Run: pip install 'transformers>=4.45'"
            ) from e

        logger.info("Loading Qwen2.5-VL: %s (quant=%s)", self.model_name, self.quantization)
        bnb_cfg = _make_bnb_config(self.quantization)

        kwargs: dict = {"device_map": "auto"}
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info("Qwen2.5-VL loaded")

    def unload(self) -> None:
        """Free model VRAM."""
        import gc
        import torch
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded Qwen-VL agent")

    def batch_describe(self, image_paths: list[Path]) -> list[str]:
        """Generate Vietnamese scene descriptions for a batch of images.

        Args:
            image_paths: List of paths to images (peak frames).

        Returns:
            List of Vietnamese description strings.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("QwenVLAgent not loaded. Call load() first.")

        import torch
        from PIL import Image

        descriptions: list[str] = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt",
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=self.temperature > 0,
                    )

                out_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                descriptions.append(text)
                logger.debug("VL described %s: %d chars", img_path.name, len(text))

            except Exception as exc:
                logger.warning("VL failed on %s: %s", img_path, exc)
                descriptions.append("")

        return descriptions
