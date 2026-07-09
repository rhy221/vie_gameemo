"""Loss functions, plus training-time augmentation for cached embeddings.

Focal Loss is recommended for emotion datasets due to class imbalance
(neutral typically dominant). Weighted CE is a simpler alternative.

Stage 1 (perception) trains on frozen, pre-extracted embeddings, not raw
pixels/audio (see extract_features.py + dataset.py cached mode) — so classic
input-level augmentation (color jitter, pitch shift, SpecAugment on a
spectrogram) can't be applied at train time without re-extracting the cache.
`embedding_augment` below is the embedding-space stand-in: Gaussian noise +
SpecAugment-style temporal masking applied directly to the cached (B, T, D)
modality tensors. Weaker than true raw-level augmentation but zero extra
extraction cost.
"""

import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def embedding_augment(
    x: Tensor,
    noise_std: float = 0.0,
    time_mask_p: float = 0.0,
    max_mask_frac: float = 0.2,
) -> Tensor:
    """Embedding-space augmentation for one modality's (B, T, D) batch.

    Two independent, additive perturbations:
      1. Gaussian noise, scaled by each sample's own embedding std (so the
         perturbation is roughly comparable across modalities/encoders with
         different typical magnitudes, instead of one fixed absolute scale).
      2. A SpecAugment-style time mask: with probability `time_mask_p` per
         sample, zero one contiguous span of up to `max_mask_frac * T` steps.
         No-op when T == 1 (nothing to mask along time).

    Args:
        x: (B, T, D) modality embeddings.
        noise_std: Relative Gaussian noise std (0 disables).
        time_mask_p: Per-sample probability of applying one time mask (0 disables).
        max_mask_frac: Max fraction of T a single mask span can cover.

    Returns:
        Augmented (B, T, D) tensor — a new tensor; `x` is not mutated.
    """
    if noise_std <= 0 and time_mask_p <= 0:
        return x

    x = x.clone()

    if noise_std > 0:
        scale = x.detach().std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x = x + torch.randn_like(x) * noise_std * scale

    if time_mask_p > 0 and x.shape[1] > 1:
        B, T, _ = x.shape
        max_len = min(max(1, int(T * max_mask_frac)), T - 1)
        for i in range(B):
            if random.random() < time_mask_p:
                mask_len = random.randint(1, max_len)
                start = random.randint(0, T - mask_len)
                x[i, start:start + mask_len, :] = 0

    return x


def modality_dropout(
    audio: Tensor, face: Tensor, context: Tensor, text: Tensor, p: float = 0.3,
    has_face: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Randomly zero one whole modality per sample (training-time augmentation).

    Forces the model to not over-rely on any single modality always being
    present. Mutates the input tensors in place (called on a batch already
    moved to device, before any grad-tracked computation).

    Args:
        audio, face, context, text: (B, T, D) per-modality batches.
        p: Per-sample probability of dropping one (randomly chosen) modality.
        has_face: Optional (B,) bool tensor passed through to fusion's own
            has_face-based masking. If given and `face` is the modality
            dropped for a sample, `has_face` is also set False there —
            otherwise the fusion's attention-mask modality gating (see
            ConvAttention4M/AttnOnly) would still assign face a nonzero
            attention weight even though its value was just zeroed here.

    Returns:
        (audio, face, context, text, has_face) — same tensors, mutated in place.
    """
    B = audio.shape[0]
    for i in range(B):
        if random.random() < p:
            drop = random.randint(0, 3)
            if drop == 0:
                audio[i] = 0
            elif drop == 1:
                face[i] = 0
                if has_face is not None:
                    has_face[i] = False
            elif drop == 2:
                context[i] = 0
            else:
                text[i] = 0
    return audio, face, context, text, has_face


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


def build_classification_criterion(
    loss_cfg,
    alpha: float | Tensor = 1.0,
) -> nn.Module:
    """Build the Stage 1/2 classification criterion from `classifier.loss` config.

    Args:
        loss_cfg: `cfg.classifier.loss` namespace (fields: type, focal.gamma, ...).
        alpha: Per-class weight tensor (from class_weights) or scalar 1.0.
            Used as FocalLoss's alpha for "focal", or CrossEntropyLoss's
            per-class `weight` (only if a Tensor) for "ce"/"weighted_ce".

    Returns:
        FocalLoss for loss.type == "focal", nn.CrossEntropyLoss otherwise.

    Raises:
        ValueError: If loss.type is not one of "ce", "weighted_ce", "focal".
    """
    loss_type = getattr(loss_cfg, "type", "focal")
    if loss_type == "focal":
        return FocalLoss(gamma=getattr(loss_cfg.focal, "gamma", 2.0), alpha=alpha)
    elif loss_type in ("ce", "weighted_ce"):
        weight = alpha if isinstance(alpha, Tensor) else None
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type!r}")


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
