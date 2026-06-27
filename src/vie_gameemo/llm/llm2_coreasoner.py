"""LLM-2: Co-Reasoner (soft token + MLP label hint → own decision).

Receives the fusion embedding (via ModalAdapter → soft token) AND the
MLP-predicted label as a hint/suggestion. The LLM reasons over the soft
token and may agree with or override the MLP label.

Requires cognition training to learn the ModalAdapter projection.
"""

import logging
from pathlib import Path

import torch

from vie_gameemo.llm.base import BaseLLMReasoner, LLMOutput
from vie_gameemo.llm.llm1_explainer import LLM1Explainer, _make_bnb_config

logger = logging.getLogger(__name__)

_VALID_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]

_COREASONER_PROMPT = """\
Classifier dự đoán streamer đang ở trạng thái: {mlp_label}.
Đây chỉ là gợi ý — có thể đúng hoặc sai.

Ngôn ngữ gốc: {source_language}
Transcript: "{transcript}"
Các nhãn có thể: {emotion_categories}

Dựa vào bằng chứng đa phương thức đã được cung cấp, hãy tự phân tích
và đưa ra nhãn cảm xúc cuối cùng. Bạn có thể đồng ý hoặc không đồng ý
với gợi ý của classifier.

Trả lời theo format:
<think>[lý luận 3-5 câu, giải thích tại sao đồng ý hoặc không đồng ý với classifier]</think>
<answer>[một nhãn từ danh sách trên]</answer>
"""


class LLM2CoReasoner(BaseLLMReasoner):
    """Co-Reasoner: soft token + MLP label hint → own prediction."""

    @staticmethod
    def build_prompt(evidence: dict) -> str:
        """Build a text prompt from evidence dict (used by LLM-4 and train_rlvr).

        Args:
            evidence: Dict with optional keys: 'face_aus', 'visual_objective',
                'audio_tone', 'transcript', 'emotion_categories'.

        Returns:
            Formatted prompt string.
        """
        categories = evidence.get(
            "emotion_categories",
            ", ".join(_VALID_LABELS),
        )
        face_aus = evidence.get("face_aus", "N/A")
        visual = evidence.get("visual_objective", "N/A")
        audio_tone = evidence.get("audio_tone", "N/A")
        transcript = evidence.get("transcript", "")
        source_language = evidence.get("source_language", "vi")

        return (
            f"Bằng chứng đa phương thức:\n"
            f"- Khuôn mặt (Action Units): {face_aus}\n"
            f"- Cảnh: {visual}\n"
            f"- Giọng nói: {audio_tone}\n"
            f"- Lời nói ({source_language}): \"{transcript}\"\n\n"
            f"Các nhãn có thể: {categories}\n\n"
            f"Dựa vào bằng chứng trên, hãy phân tích cảm xúc của streamer.\n"
            f"Trả lời theo format:\n"
            f"<think>[lý luận 3-5 câu]</think>\n"
            f"<answer>[một nhãn từ danh sách trên]</answer>"
        )

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
        """Load model + tokenizer + ModalAdapter + optional LoRA."""
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
        else:
            logger.warning("No ModalAdapter checkpoint — LLM-2 requires cognition training first")

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
        """Co-reason: consider MLP label as hint, make own decision.

        Args:
            evidence: Dict with:
                - 'fusion_emb': (1, T, 768) tensor (required)
                - 'mlp_label': MLP-predicted label (required, used as hint)
                - 'transcript': str (optional)
                - 'source_language': 'vi' | 'en' (optional)

        Returns:
            LLMOutput with LLM's own label in answer (may differ from mlp_label).
        """
        if self.model is None:
            self.load()

        if self.modal_adapter is None:
            raise RuntimeError(
                "LLM-2 requires a trained ModalAdapter. "
                "Run cognition training first (--stage cognition)."
            )

        fusion_emb = evidence["fusion_emb"].to(self.device)
        mlp_label = evidence.get("mlp_label", "neutral")

        with torch.no_grad():
            soft_token = self.modal_adapter(fusion_emb).mean(dim=1, keepdim=True)

            prompt = _COREASONER_PROMPT.format(
                mlp_label=mlp_label,
                transcript=evidence.get("transcript", ""),
                source_language=evidence.get("source_language", "vi"),
                emotion_categories=", ".join(_VALID_LABELS),
            )
            text_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            text_embeds = self.model.get_input_embeddings()(text_ids)

            # [soft_token | prompt_tokens]
            inputs_embeds = torch.cat([soft_token, text_embeds], dim=1)

            out_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            raw = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        reasoning, answer, fmt_valid = LLM1Explainer.parse_output(raw, mlp_label)

        if answer not in _VALID_LABELS:
            answer = mlp_label

        return LLMOutput(reasoning=reasoning, answer=answer, raw=raw, format_valid=fmt_valid)

    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        return [self.reason(e) for e in evidences]
