"""Loss functions: Focal Loss and weighted Cross-Entropy.

Focal Loss is recommended for emotion datasets due to class imbalance
(neutral typically dominant). Weighted CE is a simpler alternative.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Scalar or per-class weight (Tensor of shape (n_classes,)).
        gamma: Focusing parameter (higher = more focus on hard examples).
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: float | Tensor = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: (B, n_classes).
            targets: (B,) long tensor of class indices.

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        if isinstance(self.alpha, Tensor):
            alpha_t = self.alpha.to(targets.device)[targets]
        else:
            alpha_t = self.alpha
        focal = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def make_class_weights(
    labels: list[int],
    n_classes: int,
    method: str = "inverse_freq",
) -> Tensor:
    """Compute per-class weights for weighted CE.

    Args:
        labels: All training labels (integer class indices).
        n_classes: Number of classes.
        method: 'inverse_freq' | 'effective_number'.

    Returns:
        Tensor of shape (n_classes,) with weights normalized to mean=1.

    Raises:
        ValueError: If method is unknown.
    """
    counts = torch.zeros(n_classes)
    for lbl in labels:
        counts[lbl] += 1

    if method == "inverse_freq":
        counts = counts.clamp(min=1.0)
        weights = 1.0 / counts
    elif method == "effective_number":
        beta = (len(labels) - 1) / len(labels)
        effective_num = 1.0 - torch.pow(beta, counts.clamp(min=1))
        weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown class weight method: {method!r}")

    return weights / weights.mean()
