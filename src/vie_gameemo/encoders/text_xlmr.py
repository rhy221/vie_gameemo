"""Text encoder using XLM-RoBERTa-base.

Multilingual encoder that handles Vietnamese + English gaming slang
code-switching natively. PhoBERT is also a strong VN baseline, but XLM-R
handles code-switching better out of the box.

Output: token sequence (B, T, 768) where T depends on pooling.
"""

import logging

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class XLMRTextEncoder(nn.Module):
    """XLM-RoBERTa encoder for multilingual transcript embedding."""

    def __init__(
        self,
        model_name: str = "FacebookAI/xlm-roberta-base",
        max_length: int = 128,
        pooling: str = "cls",
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize text encoder.

        Args:
            model_name: HF model ID.
            max_length: Max token length (truncate longer).
            pooling: 'cls' (T=1) | 'mean' (T=1) | 'none' (T=seq_len).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = torch.device(device)

        logger.info("Loading XLM-RoBERTa: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        logger.info("Text encoder loaded and frozen")

    @torch.no_grad()
    def encode(self, text: str) -> Tensor:
        """Encode a single transcript.

        Args:
            text: Transcript string (may contain Vietnamese + English mix).
                Empty string returns zero tensor.

        Returns:
            Tensor of shape (1, T, 768):
                - T=1 for cls/mean pooling
                - T=seq_len for pooling='none'
        """
        if not text.strip():
            T = 1 if self.pooling in ("cls", "mean") else self.max_length
            return torch.zeros(1, T, 768, device=self.device)

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, seq_len, 768)
        return self._pool(hidden)

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> Tensor:
        """Batch encode transcripts with padding.

        Args:
            texts: List of transcript strings.

        Returns:
            Tensor of shape (B, T, 768).
        """
        if not texts:
            return torch.zeros(0, 1, 768, device=self.device)

        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (B, seq_len, 768)
        return self._pool(hidden)

    def _pool(self, hidden: Tensor) -> Tensor:
        """Apply configured pooling to hidden states.

        Args:
            hidden: (B, seq_len, 768).

        Returns:
            (B, T, 768) pooled tensor.
        """
        if self.pooling == "cls":
            return hidden[:, :1, :]  # CLS token → (B, 1, 768)
        elif self.pooling == "mean":
            return hidden.mean(dim=1, keepdim=True)  # (B, 1, 768)
        else:
            return hidden  # (B, seq_len, 768)
