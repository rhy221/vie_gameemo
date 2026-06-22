"""Text encoder with multiple backend support.

Default: CafeBERT (XLM-R-large + pretrained on Vietnamese, SOTA on VLUE).
Fallback: xlmr-base if T4 VRAM constrained.
Options: xlmr-large, xlm-emo (emotion-pretrain), mE5-frozen, phobert (VI-only).

Output: token sequence (B, T, D) where T depends on pooling, D depends on model.
All backends share the same encode()/encode_batch() interface.
"""

import logging

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_BACKEND_MODELS = {
    "cafebert": "uitnlp/CafeBERT",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "xlm-emo": "MilaNLProc/xlm-emo-t",
    "mE5-frozen": "intfloat/multilingual-e5-large",
}


class XLMRTextEncoder(nn.Module):
    """Multilingual text encoder (XLM-R family, CafeBERT, mE5)."""

    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        max_length: int = 128,
        pooling: str = "cls",
        freeze: bool = False,
        device: str | torch.device = "cuda",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.freeze = freeze
        self.device = torch.device(device)
        self.normalize = normalize

        logger.info("Loading text encoder: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            logger.info("Text encoder frozen")
        else:
            logger.info("Text encoder trainable (fine-tune)")

        self.d_model = self.model.config.hidden_size
        self.model = self.model.to(self.device)

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move module and keep self.device in sync with the model's params.

        The caller (e.g. extract_features.py) builds this encoder on CPU and
        then calls `.to(device)`. Without this override, self.device would stay
        "cpu" while the model moves to CUDA, sending tokenized inputs to the
        wrong device in encode()/encode_batch().
        """
        super().to(*args, **kwargs)
        self.device = next(self.model.parameters()).device
        return self

    @torch.no_grad()
    def encode(self, text: str) -> Tensor:
        """Encode a single transcript.

        Returns:
            Tensor of shape (1, T, D): T=1 for cls/mean, T=seq_len for none.
        """
        if not text.strip():
            T = 1 if self.pooling in ("cls", "mean") else self.max_length
            return torch.zeros(1, T, self.d_model, device=self.device)

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        pooled = self._pool(hidden)
        if self.normalize:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> Tensor:
        """Batch encode transcripts with padding.

        Returns:
            Tensor of shape (B, T, D).
        """
        if not texts:
            return torch.zeros(0, 1, self.d_model, device=self.device)

        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        pooled = self._pool(hidden)
        if self.normalize:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass for training (when not frozen).

        Returns:
            (B, T, D) pooled hidden states.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self._pool(hidden)
        if self.normalize:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    def _pool(self, hidden: Tensor) -> Tensor:
        if self.pooling == "cls":
            return hidden[:, :1, :]
        elif self.pooling == "mean":
            return hidden.mean(dim=1, keepdim=True)
        else:
            return hidden


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_text_encoder(encoder_cfg) -> nn.Module:
    """Build text encoder from config.

    Supports backends: cafebert, xlmr-large, xlmr-base, xlm-emo,
    mE5-frozen, phobert.

    Args:
        encoder_cfg: cfg.text_encoder (SimpleNamespace).

    Returns:
        Encoder instance (on CPU — caller moves to device).
    """
    backend = getattr(encoder_cfg, "backend", getattr(encoder_cfg, "type", "cafebert"))
    freeze = getattr(encoder_cfg, "freeze", False)
    max_length = getattr(encoder_cfg, "max_length", 128)
    pooling = getattr(encoder_cfg, "pooling", "cls")

    if backend == "phobert":
        from vie_gameemo.encoders.text_phobert import PhoBERTTextEncoder
        ph = getattr(encoder_cfg, "phobert", encoder_cfg)
        model_name = getattr(ph, "model_name", "vinai/phobert-base-v2")
        logger.info("Text encoder: PhoBERT (%s)", model_name)
        return PhoBERTTextEncoder(
            model_name=model_name,
            max_length=max_length,
            pooling=pooling,
        )

    # Resolve model name
    model_name = getattr(encoder_cfg, "model", None)
    if model_name is None:
        model_name = _BACKEND_MODELS.get(backend, _BACKEND_MODELS["cafebert"])
    backend_ns = getattr(encoder_cfg, backend.replace("-", "_"), None)
    if backend_ns is not None:
        model_name = getattr(backend_ns, "model_name", model_name)

    # Warm-start: load weights from a different pretrained model
    warm_start = getattr(encoder_cfg, "warm_start", "none")

    normalize = backend == "mE5-frozen"
    if backend == "mE5-frozen":
        freeze = True

    logger.info("Text encoder: %s (%s, freeze=%s)", backend, model_name, freeze)

    encoder = XLMRTextEncoder(
        model_name=model_name,
        max_length=max_length,
        pooling=pooling,
        freeze=freeze,
        device="cpu",
        normalize=normalize,
    )

    if warm_start and warm_start != "none":
        _apply_warm_start(encoder, warm_start)

    return encoder


def _apply_warm_start(encoder: XLMRTextEncoder, warm_start_model: str) -> None:
    """Load compatible weights from a warm-start model (e.g., xlm-emo)."""
    logger.info("Applying warm-start from %s", warm_start_model)
    try:
        warm_model = AutoModel.from_pretrained(warm_start_model)
        missing, unexpected = encoder.model.load_state_dict(
            warm_model.state_dict(), strict=False
        )
        if missing:
            logger.warning("Warm-start missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("Warm-start unexpected keys: %s", unexpected[:5])
        del warm_model
        logger.info("Warm-start applied successfully")
    except Exception as exc:
        logger.warning("Warm-start failed: %s — continuing with base weights", exc)
