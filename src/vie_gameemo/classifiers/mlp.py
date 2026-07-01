"""MLP emotion classifier.

Simple 2-layer MLP on top of the fused multimodal representation.

Input:  u_fusion of shape (B, T, 768) from Stage 3 fusion.
Output: logits of shape (B, n_classes).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EmotionClassifier(nn.Module):
    """MLP classifier head.

    Args:
        d_model: Input feature dim (default 768).
        hidden_dim: Intermediate dim.
        n_classes: Number of emotion classes.
        dropout: Dropout probability.
        pool: 'mean' | 'max' | 'cls' | 'attention'.
            'attention' learns a query vector that weights each token before
            aggregation, allowing the model to focus on emotionally peak frames.
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 256,
        n_classes: int = 8,
        dropout: float = 0.3,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.pool = pool
        self._d_model = d_model
        if pool == "attention":
            self.attn_q = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.attn_q, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self, u_fusion: Tensor, return_penultimate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Classify a fused representation.

        Args:
            u_fusion: (B, T, d_model) from Stage 3 fusion, or (B, d_model) already pooled.
            return_penultimate: If True, also return the 256-d hidden vector
                before the final classification layer (for LLM-1 faithfulness tap).

        Returns:
            Logits (B, n_classes), or (logits, penult) if return_penultimate.
        """
        if u_fusion.dim() == 3:
            if self.pool == "mean":
                h = u_fusion.mean(dim=1)
            elif self.pool == "max":
                h, _ = u_fusion.max(dim=1)
            elif self.pool == "cls":
                h = u_fusion[:, 0, :]
            elif self.pool == "attention":
                B, T, D = u_fusion.shape
                q = self.attn_q.expand(B, 1, D)                      # (B, 1, D)
                scores = (q @ u_fusion.transpose(1, 2)) * (D ** -0.5)  # (B, 1, T)
                weights = F.softmax(scores, dim=-1)                   # (B, 1, T)
                h = (weights @ u_fusion).squeeze(1)                   # (B, D)
            else:
                raise ValueError(f"Unknown pool: {self.pool!r}")
        else:
            h = u_fusion  # already (B, D)

        penult = self.net[:3](h)
        logits = self.net[3](penult)
        if return_penultimate:
            return logits, penult
        return logits
