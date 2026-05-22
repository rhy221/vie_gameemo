"""LLM-2: Co-Reasoner (modality-to-text + LLM aggregation).

Each modality is first converted to a text description:
    - Face: AU descriptions from OpenFace
    - Audio: prosody description from Qwen2-Audio
    - Visual context: scene description from Qwen2.5-VL
    - Text: transcript from Whisper

The LLM then reasons over these text descriptions and makes its own
emotion prediction (not just explanation like LLM-1).
"""

import logging

from vie_gameemo.llm.base import BaseLLMReasoner, LLMOutput
from vie_gameemo.llm.llm1_explainer import LLM1Explainer, _make_bnb_config

logger = logging.getLogger(__name__)

_COREASONER_PROMPT = """\
Bạn là chuyên gia phân tích cảm xúc. Dựa vào các mô tả đa phương thức sau:

Khuôn mặt (Action Units): {face_aus}
Bối cảnh và cảnh game: {visual_objective}
Đặc điểm giọng nói: {audio_tone}
Lời nói (transcript): "{transcript}"
Các nhãn cảm xúc có thể: {emotion_categories}

Hãy phân tích và dự đoán cảm xúc của streamer theo format:
<think>
[Lý luận 3-5 câu, kết nối bằng chứng từ nhiều modality]
</think>
<answer>[một nhãn cảm xúc từ danh sách trên]</answer>
"""

_VALID_LABELS = ["hype", "tilted", "focused", "disappointed", "shocked", "amused", "neutral"]


class LLM2CoReasoner(BaseLLMReasoner):
    """Co-Reasoner using modality-to-text inputs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quantization: str = "4bit",
        max_new_tokens: int = 400,
        temperature: float = 0.5,
        finetuned_checkpoint: str | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_name: HF model ID.
            quantization: '4bit' | '8bit' | 'none'.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature.
            finetuned_checkpoint: Optional path to LoRA adapter from SFT step.
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.finetuned_checkpoint = finetuned_checkpoint
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load model + tokenizer, with optional LoRA adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM-2: %s (quant=%s)", self.model_name, self.quantization)
        bnb_cfg = _make_bnb_config(self.quantization)
        kwargs: dict = {"device_map": "auto"}
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        else:
            kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if self.finetuned_checkpoint:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.finetuned_checkpoint)
            logger.info("Loaded LoRA adapter from %s", self.finetuned_checkpoint)

        self.model.eval()
        logger.info("LLM-2 loaded")

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reason(self, evidence: dict) -> LLMOutput:
        """Reason given modality descriptions.

        Args:
            evidence: Dict with keys:
                'face_aus', 'visual_objective', 'audio_tone', 'transcript'.
                Optionally 'emotion_categories' (list of valid label strings).

        Returns:
            LLMOutput with predicted label and reasoning.
        """
        if self.model is None:
            self.load()

        prompt = self.build_prompt(evidence)
        raw = self._generate(prompt)
        reasoning, answer, fmt_valid = LLM1Explainer.parse_output(raw)
        return LLMOutput(reasoning=reasoning, answer=answer, raw=raw, format_valid=fmt_valid)

    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        """Batch reason (sequential).

        Args:
            evidences: List of evidence dicts.

        Returns:
            List of LLMOutput.
        """
        return [self.reason(e) for e in evidences]

    def _generate(self, prompt: str) -> str:
        """Run text generation for a single prompt."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @staticmethod
    def build_prompt(evidence: dict) -> str:
        """Build the reasoning prompt from evidence dict.

        Args:
            evidence: Dict with modality description fields.

        Returns:
            Formatted prompt string.
        """
        categories = evidence.get("emotion_categories", _VALID_LABELS)
        face_aus = evidence.get("face_aus", "N/A")
        if isinstance(face_aus, dict):
            face_aus = ", ".join(f"AU{k}={v:.1f}" for k, v in face_aus.items()) or "N/A"
        return _COREASONER_PROMPT.format(
            face_aus=face_aus,
            visual_objective=evidence.get("visual_objective", "N/A"),
            audio_tone=evidence.get("audio_tone", "N/A"),
            transcript=evidence.get("transcript", ""),
            emotion_categories=", ".join(categories),
        )
