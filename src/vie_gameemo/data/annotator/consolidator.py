"""Consolidator: merge all per-modality descriptions into coherent reasoning.

Uses Qwen2.5-Instruct (7B / 32B / 72B depending on compute) to take the
outputs of all upstream agents (Cved, Cvod, Catd, Cls) along with the
ground-truth emotion label, and produce a structured reasoning explanation
in Vietnamese.

Output format:
    <think>
    [3-5 sentences linking modality evidence to the emotion]
    </think>
    <answer>{emotion_label}</answer>
"""

import logging
import re

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT_TEMPLATE = """\
Bạn là chuyên gia phân tích cảm xúc đa phương thức. Viết đoạn reasoning ngắn
(3-5 câu) bằng tiếng Việt, giải thích vì sao streamer này đang trong trạng
thái {emotion_label}.

Ngôn ngữ gốc của clip: {source_language}

Bằng chứng đa phương thức:
- Khuôn mặt (Action Units): {face_aus}
- Cảnh và bối cảnh: {visual_objective}
- Đặc điểm giọng nói: {audio_tone}
- Lời nói (transcript): "{transcript}"

Hướng dẫn:
- Transcript có thể bằng tiếng Việt HOẶC tiếng Anh — hãy hiểu trực tiếp,
  KHÔNG dịch transcript trong câu trả lời
- Trả lời hoàn toàn bằng tiếng Việt
- Liên kết các bằng chứng từ nhiều modality
- Nếu có slang gaming tiếng Anh, hãy giải thích nghĩa
- KHÔNG bịa thông tin không có trong bằng chứng

Output theo format CHÍNH XÁC:
<think>
[Lý luận của bạn ở đây]
</think>
<answer>{emotion_label}</answer>
"""

_FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


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


class Consolidator:
    """Multi-modal evidence consolidator using a large Qwen2.5 model."""

    def __init__(
        self,
        model_name: str,
        quantization: str = "4bit",
        max_new_tokens: int = 400,
        temperature: float = 0.5,
    ) -> None:
        """Initialize consolidator.

        Args:
            model_name: HF model ID (e.g., "Qwen/Qwen2.5-7B-Instruct").
            quantization: "4bit" | "8bit" | "none".
            max_new_tokens: Max reasoning length.
            temperature: Sampling temperature.
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load Qwen2.5-Instruct model + tokenizer.

        Raises:
            ImportError: If transformers not installed.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading consolidator: %s (quant=%s)", self.model_name, self.quantization)
        bnb_cfg = _make_bnb_config(self.quantization)

        kwargs: dict = {"device_map": "auto"}
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        else:
            kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()
        logger.info("Consolidator loaded")

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def consolidate(
        self,
        emotion_label: str,
        face_aus: dict[str, float],
        visual_objective: str,
        audio_tone: str,
        transcript: str,
        source_language: str = "vi",
    ) -> str:
        """Generate structured reasoning for one clip.

        Args:
            emotion_label: Ground-truth emotion (from manual annotation).
            face_aus: AU code → intensity dict.
            visual_objective: Cvod from Qwen-VL.
            audio_tone: Catd from Qwen-Audio.
            transcript: Cls from Whisper.
            source_language: "vi" or "en" — passed to prompt for context.

        Returns:
            Reasoning text in <think>...</think><answer>...</answer> format.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If output format invalid after retry.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Consolidator not loaded. Call load() first.")

        face_aus_str = ", ".join(f"AU{k}={v:.1f}" for k, v in face_aus.items())
        prompt = CONSOLIDATION_PROMPT_TEMPLATE.format(
            emotion_label=emotion_label,
            face_aus=face_aus_str or "N/A",
            visual_objective=visual_objective or "N/A",
            audio_tone=audio_tone or "N/A",
            transcript=transcript or "",
            source_language=source_language,
        )

        for attempt in range(2):
            output = self._generate(prompt)
            if self.validate_output_format(output):
                return output
            if attempt == 0:
                logger.warning("Consolidator output format invalid; retrying")

        logger.warning("Could not get valid format after retry; returning raw output")
        return output

    def batch_consolidate(self, inputs: list[dict]) -> list[str]:
        """Batch consolidation. Each input dict has the keys of consolidate().

        Args:
            inputs: List of dicts with keys: emotion_label, face_aus,
                visual_objective, audio_tone, transcript.

        Returns:
            List of reasoning strings.
        """
        return [self.consolidate(**inp) for inp in inputs]

    def _generate(self, prompt: str) -> str:
        """Run text generation for a single prompt.

        Args:
            prompt: Full user prompt string.

        Returns:
            Generated text (decoded, no special tokens).
        """
        import torch

        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        out_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    @staticmethod
    def validate_output_format(text: str) -> bool:
        """Check if text contains both <think>...</think> and <answer>...</answer>.

        Args:
            text: Generated text to validate.

        Returns:
            True if format matches expected structure.
        """
        return bool(_FORMAT_PATTERN.search(text))
