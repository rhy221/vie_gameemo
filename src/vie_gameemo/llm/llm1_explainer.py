"""LLM-1: Post-hoc Explainer (no training, prompt-only).

The classifier (Stage 4) predicts the emotion label. The LLM then explains
WHY, given the multimodal evidence. No fine-tuning needed; just prompt
engineering with a frozen Qwen2.5-7B.

This is the cheapest setup (no training compute) and serves as the baseline
for LLM ablation.
"""

import logging
import re

from vie_gameemo.llm.base import BaseLLMReasoner, LLMOutput

logger = logging.getLogger(__name__)

_FORMAT_REMINDER = (
    "\n\nQuan trọng: Trả lời CHÍNH XÁC theo format:\n"
    "<think>[lý luận]</think>\n<answer>[nhãn cảm xúc]</answer>"
)


def _make_bnb_config(quantization: str):
    """Build BitsAndBytesConfig."""
    try:
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        pass
    return None


class LLM1Explainer(BaseLLMReasoner):
    """Post-hoc explainer using prompt-only Qwen2.5-Instruct."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        prompt_template: str = "",
        quantization: str = "4bit",
        max_new_tokens: int = 300,
        temperature: float = 0.7,
    ) -> None:
        """Initialize.

        Args:
            model_name: HF model ID.
            prompt_template: Vietnamese template with format fields.
            quantization: '4bit' | '8bit' | 'none'.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature.
        """
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM-1: %s (quant=%s)", self.model_name, self.quantization)
        bnb_cfg = _make_bnb_config(self.quantization)
        kwargs: dict = {"device_map": "auto"}
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        else:
            kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()
        logger.info("LLM-1 loaded")

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
        """Generate explanation for a single sample.

        Args:
            evidence: Dict with keys: 'label', 'face_aus', 'game_context',
                'pitch_hz', 'rms_db', 'shout', 'transcript'.

        Returns:
            LLMOutput with reasoning and predicted emotion.
        """
        if self.model is None:
            self.load()

        prompt = self.prompt_template.format(
            label=evidence.get("label", "unknown"),
            face_aus=evidence.get("face_aus", "N/A"),
            game_context=evidence.get("game_context", "N/A"),
            pitch_hz=evidence.get("pitch_hz", 0),
            rms_db=evidence.get("rms_db", 0),
            shout=evidence.get("shout", False),
            transcript=evidence.get("transcript", ""),
        )

        raw = self._generate(prompt)
        reasoning, answer, fmt_valid = self.parse_output(raw, evidence.get("label", "neutral"))

        if not fmt_valid:
            logger.debug("LLM-1 output format invalid; retrying with reminder")
            raw = self._generate(prompt + _FORMAT_REMINDER)
            reasoning, answer, fmt_valid = self.parse_output(raw, evidence.get("label", "neutral"))

        return LLMOutput(reasoning=reasoning, answer=answer, raw=raw, format_valid=fmt_valid)

    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        """Batch reason (sequential for now).

        Args:
            evidences: List of evidence dicts.

        Returns:
            List of LLMOutput.
        """
        return [self.reason(e) for e in evidences]

    def _generate(self, prompt: str) -> str:
        """Run generation for a single prompt string."""
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
    def parse_output(raw: str, fallback_label: str = "neutral") -> tuple[str, str, bool]:
        """Parse <think>...</think><answer>...</answer> from raw text.

        Args:
            raw: Generated text.
            fallback_label: Label to use if <answer> tag missing.

        Returns:
            Tuple of (reasoning, answer, format_valid).
        """
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else raw.strip()
        answer = answer_match.group(1).strip() if answer_match else fallback_label
        format_valid = bool(think_match and answer_match)
        return reasoning, answer, format_valid
