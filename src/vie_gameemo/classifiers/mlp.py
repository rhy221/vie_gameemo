"""MLP emotion classifier.

Simple 2-layer MLP on top of the fused multimodal representation.

Input:  u_fusion of shape (B, T, 768) from Stage 3 fusion.
Output: logits of shape (B, n_classes).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _pool(u_fusion: Tensor, pool: str, attn_score: nn.Module | None = None) -> Tensor:
    """Pool (B, T, D) → (B, D) over the T (time/token) axis. Pass-through if already (B, D)."""
    if u_fusion.dim() != 3:
        return u_fusion
    if pool == "mean":
        return u_fusion.mean(dim=1)
    elif pool == "max":
        h, _ = u_fusion.max(dim=1)
        return h
    elif pool == "cls":
        return u_fusion[:, 0, :]
    elif pool == "attention":
        weights = torch.softmax(attn_score(u_fusion), dim=1)  # (B, T, 1)
        return (weights * u_fusion).sum(dim=1)
    else:
        raise ValueError(f"Unknown pool: {pool!r}")


class EmotionClassifier(nn.Module):
    """MLP classifier head.

    Args:
        d_model: Input feature dim (default 768).
        hidden_dim: Intermediate dim.
        n_classes: Number of emotion classes.
        dropout: Dropout probability.
        pool: 'mean' | 'max' | 'cls' (uses first token) | 'attention' (learned weighted sum).
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
        if pool == "attention":
            self.attn_score = nn.Linear(d_model, 1)
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
        h = _pool(u_fusion, self.pool, getattr(self, "attn_score", None))
        penult = self.net[:3](h)
        logits = self.net[3](penult)
        if return_penultimate:
            return logits, penult
        return logits


class HierarchicalEmotionClassifier(nn.Module):
    """2-stage hierarchical classifier: coarse easy/hard group, then within-group.

    Motivated by test-set confusion-matrix analysis on this project's gaming_8
    label set: {shocked, fear, tilted, disgusted} form a mutually-confusable
    "hard" cluster (F1 0.62-0.76), clearly separated from {neutral, hype,
    amused, sad} ("easy" cluster, F1 0.87-0.94) which rarely get confused with
    the hard cluster or each other. Stage A picks the group; Stage B has one
    small head per group specializing in disambiguating within it.

    The combined per-class distribution is
        P(class) = P(group of class) * P(class | group),
    so log P(class) already sums to 1 across all n_classes. This can be fed
    directly as "logits" into the existing loss functions (FocalLoss /
    CrossEntropyLoss) unmodified: log_softmax is idempotent on values that
    already form a valid log-probability distribution (logsumexp = log(1) = 0),
    so their internal softmax/log_softmax is a no-op here.

    Args:
        d_model: Input feature dim (default 768).
        hidden_dim: Shared trunk hidden dim.
        n_classes: Total number of emotion classes (must equal len(easy_idx)+len(hard_idx)).
        dropout: Dropout probability (shared trunk only).
        pool: 'mean' | 'max' | 'cls' | 'attention' (see `_pool`).
        easy_idx: Class indices belonging to the "easy" group.
        hard_idx: Class indices belonging to the "hard" group.
            `easy_idx` and `hard_idx` together must partition `range(n_classes)`
            exactly (each class index in exactly one group).
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 256,
        n_classes: int = 8,
        dropout: float = 0.3,
        pool: str = "mean",
        easy_idx: tuple[int, ...] = (0, 1, 2, 4),
        hard_idx: tuple[int, ...] = (3, 5, 6, 7),
    ) -> None:
        super().__init__()
        covered = sorted(list(easy_idx) + list(hard_idx))
        if covered != list(range(n_classes)):
            raise ValueError(
                f"easy_idx={easy_idx} + hard_idx={hard_idx} must partition "
                f"range(n_classes={n_classes}) exactly; got covered={covered}"
            )
        self.pool = pool
        self.n_classes = n_classes
        if pool == "attention":
            self.attn_score = nn.Linear(d_model, 1)

        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.group_head = nn.Linear(hidden_dim, 2)  # 0=easy, 1=hard
        self.easy_head = nn.Linear(hidden_dim, len(easy_idx))
        self.hard_head = nn.Linear(hidden_dim, len(hard_idx))

        self.register_buffer("easy_idx", torch.tensor(easy_idx, dtype=torch.long))
        self.register_buffer("hard_idx", torch.tensor(hard_idx, dtype=torch.long))

    def forward(
        self, u_fusion: Tensor, return_penultimate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Classify a fused representation via the easy/hard hierarchy.

        Args:
            u_fusion: (B, T, d_model) from Stage 3 fusion, or (B, d_model) already pooled.
            return_penultimate: If True, also return the shared trunk's
                hidden_dim-d vector (for LLM-1 faithfulness tap — same role
                and shape as `EmotionClassifier`'s penultimate).

        Returns:
            Log-probabilities shaped like logits (B, n_classes) — safe to use
            directly with FocalLoss/CrossEntropyLoss (see class docstring),
            or (logits, penult) if return_penultimate.
        """
        h = _pool(u_fusion, self.pool, getattr(self, "attn_score", None))
        penult = self.trunk(h)  # (B, hidden_dim)

        group_log_probs = F.log_softmax(self.group_head(penult), dim=-1)  # (B, 2)
        easy_log_probs = F.log_softmax(self.easy_head(penult), dim=-1)    # (B, n_easy)
        hard_log_probs = F.log_softmax(self.hard_head(penult), dim=-1)    # (B, n_hard)

        # dtype follows group_log_probs (not penult): under autocast, log_softmax
        # is promoted to float32 for numerical stability even when penult itself
        # is bf16/fp16 — allocating from penult's dtype would mismatch what's
        # actually written below and fail on the indexed assignment.
        B = penult.shape[0]
        logits = group_log_probs.new_zeros(B, self.n_classes)
        logits[:, self.easy_idx] = group_log_probs[:, 0:1] + easy_log_probs
        logits[:, self.hard_idx] = group_log_probs[:, 1:2] + hard_log_probs

        if return_penultimate:
            return logits, penult
        return logits
