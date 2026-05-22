"""MLP emotion classifier.

Simple 2-layer MLP on top of the fused multimodal representation.

Input:  u_fusion of shape (B, T, 768) from Stage 3 fusion.
Output: logits of shape (B, n_classes).
"""

import torch
from torch import Tensor, nn


class EmotionClassifier(nn.Module):
    """MLP classifier head.

    Args:
        d_model: Input feature dim (default 768).
        hidden_dim: Intermediate dim.
        n_classes: Number of emotion classes.
        dropout: Dropout probability.
        pool: 'mean' | 'max' | 'cls' (uses first token).
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 256,
        n_classes: int = 7,
        dropout: float = 0.3,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, u_fusion: Tensor) -> Tensor:
        """Classify a fused representation.

        Args:
            u_fusion: (B, T, d_model) from Stage 3 fusion, or (B, d_model) already pooled.

        Returns:
            Logits of shape (B, n_classes).
        """
        if u_fusion.dim() == 3:
            if self.pool == "mean":
                h = u_fusion.mean(dim=1)
            elif self.pool == "max":
                h, _ = u_fusion.max(dim=1)
            elif self.pool == "cls":
                h = u_fusion[:, 0, :]
            else:
                raise ValueError(f"Unknown pool: {self.pool!r}")
        else:
            h = u_fusion  # already (B, D)

        return self.net(h)
