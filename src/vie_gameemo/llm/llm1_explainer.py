"""LLM-1: Post-hoc Explainer.

Two modes:
  1. Zero-shot (no training): prompt with label + transcript → explanation.
  2. Trained (with explanation checkpoint): soft tokens including MLP
     penultimate vector → generates "Cues: ... Emotion: ..." format.

In both modes, LLM-1 never overrides the MLP label.
"""

import logging
import re

import torch

from vie_gameemo.llm.base import BaseLLMReasoner, LLMOutput

logger = logging.getLogger(__name__)

_EXPLAIN_PROMPT = """\
Streamer được dự đoán đang ở trạng thái: {label}.
Ngôn ngữ gốc của clip: {source_language}
Lời nói (transcript): "{transcript}"

Transcript có thể bằng tiếng Việt hoặc tiếng Anh — hiểu trực tiếp, KHÔNG dịch.
Trả lời hoàn toàn bằng tiếng Việt.

Hãy giải thích ngắn gọn (dưới 100 từ) vì sao streamer đang ở trạng thái này,
dựa trên bằng chứng đa phương thức đã được cung cấp.

Trả lời theo format:
<think>[lý luận]</think>
<answer>{label}</answer>
"""

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


_CUE_PROMPT = (
    "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được "
    "và xác định cảm xúc.\n"
    "Trả lời theo format: Cues: face: ...; voice: ...; scene: ...; text: .... "
    "Emotion: [nhãn]."
)


class LLM1Explainer(BaseLLMReasoner):
    """Post-hoc explainer: soft token + MLP label → explanation.

    Supports both zero-shot (no training) and trained mode (with
    explanation checkpoint containing proj_penult weights).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        prompt_template: str = "",
        quantization: str = "4bit",
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        modal_adapter_ckpt: str | None = None,
        d_fusion: int = 768,
        d_penult: int = 256,
    ) -> None:
        self.model_name = model_name
        self.prompt_template = prompt_template or _EXPLAIN_PROMPT
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.modal_adapter_ckpt = modal_adapter_ckpt
        self.d_fusion = d_fusion
        self.d_penult = d_penult
        self.model = None
        self.tokenizer = None
        self.modal_adapter = None
        self._trained_mode = False

    def load(self) -> None:
        """Load model, tokenizer, and optional pre-trained ModalAdapter."""
        from pathlib import Path

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

        if self.modal_adapter_ckpt and Path(self.modal_adapter_ckpt).exists():
            from vie_gameemo.llm.modal_adapter import ModalAdapter

            ckpt_data = torch.load(
                Path(self.modal_adapter_ckpt), map_location="cpu", weights_only=True,
            )
            self._trained_mode = "llm_adapter" in ckpt_data and \
                any("proj_penult" in k for k in ckpt_data.get("llm_adapter", {}).keys())

            llm_hidden = self.model.config.hidden_size
            self.modal_adapter = ModalAdapter.from_checkpoint(
                Path(self.modal_adapter_ckpt),
                d_fusion=self.d_fusion, d_llm=llm_hidden,
                d_penult=self.d_penult,
            ).to(self.model.device)
            self.modal_adapter.eval()

            if self._trained_mode:
                logger.info("Loaded trained ModalAdapter (with penult) from %s", self.modal_adapter_ckpt)
            else:
                logger.info("Loaded ModalAdapter from %s", self.modal_adapter_ckpt)

            # Load LoRA weights if present (Stage B checkpoint)
            if ckpt_data.get("llm_peft") is not None:
                try:
                    from peft import LoraConfig, get_peft_model
                    lora_config = LoraConfig(
                        r=4, lora_alpha=8,
                        target_modules=["q_proj", "v_proj"],
                        bias="none", task_type="CAUSAL_LM",
                    )
                    self.model = get_peft_model(self.model, lora_config)
                    self.model.load_state_dict(ckpt_data["llm_peft"], strict=False)
                    self.model.eval()
                    logger.info("Loaded LLM LoRA weights from checkpoint")
                except Exception as e:
                    logger.warning("Failed to load LoRA weights: %s", e)

        logger.info("LLM-1 loaded (trained_mode=%s)", self._trained_mode)

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        self.model = None
        self.tokenizer = None
        self.modal_adapter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reason(self, evidence: dict) -> LLMOutput:
        """Generate explanation for a classifier prediction.

        Args:
            evidence: Dict with:
                - 'label': MLP-predicted label (required)
                - 'fusion_emb': (1, T, 768) tensor (optional, for soft token)
                - 'penult': (1, 256) MLP penultimate vector (optional, trained mode)
                - 'transcript': str (optional)
                - 'source_language': 'vi' | 'en' (optional)

        Returns:
            LLMOutput. answer always equals the input label (LLM-1 never overrides).
        """
        if self.model is None:
            self.load()

        label = evidence.get("label", "neutral")
        fusion_emb = evidence.get("fusion_emb")

        if fusion_emb is not None and self.modal_adapter is not None:
            raw = self._generate_with_soft_token(fusion_emb, label, evidence)
        else:
            raw = self._generate_text_only(label, evidence)

        reasoning, _, fmt_valid = self.parse_output(raw, label)

        if not fmt_valid:
            raw = self._generate_text_only(label, evidence, append=_FORMAT_REMINDER)
            reasoning, _, fmt_valid = self.parse_output(raw, label)

        return LLMOutput(reasoning=reasoning, answer=label, raw=raw, format_valid=fmt_valid)

    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        return [self.reason(e) for e in evidences]

    def _generate_with_soft_token(
        self, fusion_emb: torch.Tensor, label: str, evidence: dict,
    ) -> str:
        """Generate conditioned on soft token + text prompt with label."""
        fusion_emb = fusion_emb.to(self.model.device)
        penult = evidence.get("penult")
        if penult is not None:
            penult = penult.to(self.model.device)

        with torch.no_grad():
            soft_tokens, soft_mask = self.modal_adapter(
                fusion_emb,
                penult=penult,
                audio=evidence.get("audio_emb"),
                face=evidence.get("face_emb"),
                context=evidence.get("context_emb"),
                text=evidence.get("text_emb"),
                has_face=evidence.get("has_face"),
            )

            if self._trained_mode:
                prompt = _CUE_PROMPT
            else:
                prompt = self.prompt_template.format(
                    label=label,
                    transcript=evidence.get("transcript", ""),
                    source_language=evidence.get("source_language", "vi"),
                )
            text_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            text_embeds = self.model.get_input_embeddings()(text_ids)

            inputs_embeds = torch.cat([soft_tokens, text_embeds], dim=1)

            out_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    def _generate_text_only(
        self, label: str, evidence: dict, append: str = "",
    ) -> str:
        """Fallback: text-only prompt when no ModalAdapter available."""
        prompt = self.prompt_template.format(
            label=label,
            transcript=evidence.get("transcript", ""),
            source_language=evidence.get("source_language", "vi"),
        ) + append

        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
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
    def parse_output(raw: str, fallback_label: str = "neutral") -> tuple[str, str, bool]:
        """Parse output from either format.

        Tries "Cues: ... Emotion: ..." first (trained mode), then falls
        back to "<think>...</think><answer>...</answer>" (zero-shot mode).
        """
        # Trained format: "Cues: ... Emotion: label."
        cues_match = re.search(r"Cues:\s*(.*?)(?:\.\s*Emotion:)", raw, re.DOTALL)
        emotion_match = re.search(r"Emotion:\s*(\w+)", raw)
        if cues_match and emotion_match:
            reasoning = cues_match.group(1).strip()
            answer = emotion_match.group(1).strip().lower()
            return reasoning, answer, True

        # Zero-shot format: <think>...</think><answer>...</answer>
        think_matches = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        reasoning = think_matches[-1].strip() if think_matches else raw.strip()
        answer = answer_match.group(1).strip() if answer_match else fallback_label
        format_valid = bool(think_matches and answer_match)
        return reasoning, answer, format_valid
