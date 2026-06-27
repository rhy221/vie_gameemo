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

    def batch_describe(
        self,
        image_paths: list[Path],
        prompt_override: str | None = None,
    ) -> list[str]:
        """Generate Vietnamese scene descriptions for a batch of images.

        Args:
            image_paths: List of paths to images.
            prompt_override: Use this prompt instead of self.prompt.

        Returns:
            List of Vietnamese description strings.
        """
        from PIL import Image
        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(None)
        return self._describe_images(images, prompt_override, names=[p.name for p in image_paths])

    def batch_describe_images(
        self,
        images: list,
        prompt_override: str | None = None,
    ) -> list[str]:
        """Generate descriptions from PIL Images or numpy arrays directly.

        Args:
            images: List of PIL.Image or numpy arrays (BGR).
            prompt_override: Use this prompt instead of self.prompt.

        Returns:
            List of Vietnamese description strings.
        """
        from PIL import Image
        import numpy as np
        pil_images = []
        for img in images:
            if img is None:
                pil_images.append(None)
            elif isinstance(img, np.ndarray):
                import cv2
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                pil_images.append(None)
        return self._describe_images(pil_images, prompt_override)

    def _describe_images(
        self,
        images: list,
        prompt_override: str | None = None,
        names: list[str] | None = None,
    ) -> list[str]:
        """Core description loop over PIL images."""
        if self.model is None or self.processor is None:
            raise RuntimeError("QwenVLAgent not loaded. Call load() first.")

        import torch

        prompt = prompt_override or self.prompt
        descriptions: list[str] = []

        for i, image in enumerate(images):
            name = names[i] if names else f"image_{i}"
            if image is None:
                descriptions.append("")
                continue
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
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
                logger.debug("VL described %s: %d chars", name, len(text))

            except Exception as exc:
                logger.warning("VL failed on %s: %s", name, exc)
                descriptions.append("")

        return descriptions
