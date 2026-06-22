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


def fused_mixup(
    embeddings: Tensor,
    labels: Tensor,
    n_classes: int,
    alpha: float = 0.4,
    rare_classes: list[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Manifold Mixup at fused-embedding level for class imbalance.

    Mixes pairs of samples in the fused embedding space. When rare_classes
    is specified, at least one sample in each pair is from a rare class.

    Args:
        embeddings: (B, T, D) fused embeddings.
        labels: (B,) integer class labels.
        n_classes: Total number of classes.
        alpha: Beta distribution parameter for mixup ratio.
        rare_classes: Class indices to oversample in mixing pairs.

    Returns:
        (mixed_embeddings, labels_a, labels_b, lam) for mixup loss:
            loss = lam * criterion(logits, labels_a) + (1-lam) * criterion(logits, labels_b)
    """
    import numpy as np

    B = embeddings.size(0)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)

    if rare_classes and len(rare_classes) > 0:
        rare_mask = torch.zeros(B, dtype=torch.bool, device=labels.device)
        for rc in rare_classes:
            rare_mask |= (labels == rc)
        rare_idx = rare_mask.nonzero(as_tuple=True)[0]
        if len(rare_idx) > 0:
            perm = rare_idx[torch.randint(len(rare_idx), (B,), device=labels.device)]
        else:
            perm = torch.randperm(B, device=embeddings.device)
    else:
        perm = torch.randperm(B, device=embeddings.device)

    mixed = lam * embeddings + (1 - lam) * embeddings[perm]
    return mixed, labels, labels[perm], lam


def per_class_metrics(
    preds: list[int],
    labels: list[int],
    n_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """Compute per-class F1, recall, precision + confusion matrix.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.
        n_classes: Number of classes.
        class_names: Optional names for each class index.

    Returns:
        Dict with 'per_class_f1', 'per_class_recall', 'per_class_precision',
        'confusion_matrix', 'macro_f1'.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix as sk_confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    present_labels = sorted(set(labels) | set(preds))

    f1_per = f1_score(labels, preds, labels=present_labels, average=None, zero_division=0)
    recall_per = recall_score(labels, preds, labels=present_labels, average=None, zero_division=0)
    precision_per = precision_score(labels, preds, labels=present_labels, average=None, zero_division=0)

    per_class_f1 = {}
    per_class_recall = {}
    per_class_precision = {}
    for i, cls_idx in enumerate(present_labels):
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        per_class_f1[name] = float(f1_per[i])
        per_class_recall[name] = float(recall_per[i])
        per_class_precision[name] = float(precision_per[i])

    cm = sk_confusion_matrix(labels, preds, labels=list(range(n_classes)))

    return {
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "confusion_matrix": cm.tolist(),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
