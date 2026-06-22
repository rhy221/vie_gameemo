"""LLM-2: Co-Reasoner (modality-to-text + LLM aggregation).

Each modality is first converted to a text description:
    - Face: AU descriptions from OpenFace
    - Audio: prosody description from Qwen2-Audio
    - Visual context: scene description from Qwen2.5-VL
    - Text: transcript from Whisper

The LLM then reasons over these text descriptions and makes its own
emotion prediction (not just explanation like LLM-1).

Annotation-free path: when evidence contains 'fusion_emb' (a (1, T, 768)
tensor) and a cognition checkpoint is provided, ModalAdapter projects the
embedding into the LLM space as a soft token, bypassing text evidence.
"""

import logging
from pathlib import Path

import torch

from vie_gameemo.llm.base import BaseLLMReasoner, LLMOutput
from vie_gameemo.llm.llm1_explainer import LLM1Explainer, _make_bnb_config

logger = logging.getLogger(__name__)

_COREASONER_PROMPT = """\
Bạn là chuyên gia phân tích cảm xúc. Dựa vào các mô tả đa phương thức sau:

Ngôn ngữ gốc của clip: {source_language}

Khuôn mặt (Action Units): {face_aus}
Bối cảnh và cảnh game: {visual_objective}
Đặc điểm giọng nói: {audio_tone}
Lời nói (transcript): "{transcript}"
Các nhãn cảm xúc có thể: {emotion_categories}

Transcript có thể bằng tiếng Việt hoặc tiếng Anh — hiểu trực tiếp, KHÔNG dịch.
Trả lời hoàn toàn bằng tiếng Việt.

Hãy phân tích và dự đoán cảm xúc của streamer theo format:
<think>
[Lý luận 3-5 câu, kết nối bằng chứng từ nhiều modality]
</think>
<answer>[một nhãn cảm xúc từ danh sách trên]</answer>
"""

_VALID_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


class LLM2CoReasoner(BaseLLMReasoner):
    """Co-Reasoner using modality-to-text inputs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quantization: str = "4bit",
        max_new_tokens: int = 400,
        temperature: float = 0.5,
        finetuned_checkpoint: str | None = None,
        modal_adapter_ckpt: Path | None = None,
        d_fusion: int = 768,
    ) -> None:
        """Initialize.

        Args:
            model_name: HF model ID.
            quantization: '4bit' | '8bit' | 'none'.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature.
            finetuned_checkpoint: Optional path to LoRA adapter from SFT step.
            modal_adapter_ckpt: Path to cognition checkpoint containing 'llm_adapter'
                weights. When provided, enables annotation-free inference via soft tokens.
            d_fusion: Fusion embedding dim (768 by default, matches ConvAttention4M output).
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.finetuned_checkpoint = finetuned_checkpoint
        self.modal_adapter_ckpt = Path(modal_adapter_ckpt) if modal_adapter_ckpt else None
        self.d_fusion = d_fusion
        self.model = None
        self.tokenizer = None
        self.modal_adapter = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        if self.modal_adapter_ckpt is not None and self.modal_adapter_ckpt.exists():
            from vie_gameemo.llm.modal_adapter import ModalAdapter
            llm_hidden = self.model.config.hidden_size
            self.modal_adapter = ModalAdapter.from_checkpoint(
                self.modal_adapter_ckpt, d_fusion=self.d_fusion, d_llm=llm_hidden
            ).to(self.device)
            self.modal_adapter.eval()
            logger.info("Loaded ModalAdapter from %s", self.modal_adapter_ckpt)

        logger.info("LLM-2 loaded")

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
        """Reason given evidence.

        Dispatches to annotation-free path when evidence contains 'fusion_emb'
        and modal adapter is loaded; otherwise uses text evidence path.

        Args:
            evidence: Dict with either:
                - Text path: 'face_aus', 'visual_objective', 'audio_tone', 'transcript'
                - Embedding path: 'fusion_emb' (Tensor, shape (1, T, 768))
                Optionally 'emotion_categories'.

        Returns:
            LLMOutput with predicted label and reasoning.
        """
        if self.model is None:
            self.load()

        if "fusion_emb" in evidence and self.modal_adapter is not None:
            return self._reason_with_embeddings(evidence["fusion_emb"])
        return self._reason_with_text(evidence)

    def _reason_with_text(self, evidence: dict) -> LLMOutput:
        """Text evidence path (requires annotation JSON fields)."""
        prompt = self.build_prompt(evidence)
        raw = self._generate(prompt)
        reasoning, answer, fmt_valid = LLM1Explainer.parse_output(raw)
        return LLMOutput(reasoning=reasoning, answer=answer, raw=raw, format_valid=fmt_valid)

    def _reason_with_embeddings(self, fusion_emb: torch.Tensor) -> LLMOutput:
        """Annotation-free path: inject fusion embedding as soft token via modal adapter.

        Args:
            fusion_emb: (1, T, d_fusion) fused embedding from ConvAttention4M.

        Returns:
            LLMOutput from soft-token-conditioned generation.
        """
        fusion_emb = fusion_emb.to(self.device)
        with torch.no_grad():
            soft_token = self.modal_adapter(fusion_emb).mean(dim=1, keepdim=True)  # (1, 1, H)

            instruction = (
                "Dựa trên đặc trưng đa phương thức của clip game, hãy phân tích "
                "và xác định cảm xúc của streamer. Trả lời theo định dạng "
                "<think>[lý luận]</think><answer>[nhãn]</answer>."
            )
            text_ids = self.tokenizer.encode(instruction, return_tensors="pt").to(self.device)
            text_embeds = self.model.get_input_embeddings()(text_ids)  # (1, L, H)

            inputs_embeds = torch.cat([soft_token, text_embeds], dim=1)  # (1, 1+L, H)

            out_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            raw = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        reasoning, answer, fmt_valid = LLM1Explainer.parse_output(raw, "neutral")
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
            source_language=evidence.get("source_language", "vi"),
        )
