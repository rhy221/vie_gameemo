"""Stage 4 emotion classifier (head on top of fused representation)."""

from torch import nn


def get_classifier(ccfg, d_model: int, device, classifier_type: str | None = None) -> nn.Module:
    """Factory: build the classifier head from `cfg.classifier`.

    Centralizes the EmotionClassifier vs HierarchicalEmotionClassifier choice
    so every call site (train/eval/inference) builds the exact same
    architecture from one config flag — avoids a checkpoint/architecture
    mismatch (same class of bug as fusion_type drift; see fusion.get_fusion).

    Args:
        ccfg: `cfg.classifier` namespace.
        d_model: Fusion output dim (`cfg.fusion.d_model`).
        device: Torch device to move the module to.
        classifier_type: Optional explicit override ("flat" | "hierarchical"),
            e.g. from `infer_classifier_type_from_checkpoint`. Takes priority
            over `ccfg.hierarchical.enabled` — pass this whenever rebuilding
            a classifier to load an existing checkpoint into.

    Returns:
        `HierarchicalEmotionClassifier` if `classifier_type == "hierarchical"`,
        or (when `classifier_type` is None) if `ccfg.hierarchical.enabled` is
        True; else the flat `EmotionClassifier` (default / revert path).
    """
    from vie_gameemo.classifiers.mlp import EmotionClassifier, HierarchicalEmotionClassifier

    hcfg = getattr(ccfg, "hierarchical", None)
    if classifier_type is not None:
        use_hierarchical = classifier_type == "hierarchical"
    else:
        use_hierarchical = hcfg is not None and getattr(hcfg, "enabled", False)

    if use_hierarchical:
        return HierarchicalEmotionClassifier(
            d_model=d_model,
            hidden_dim=ccfg.hidden_dim,
            n_classes=ccfg.n_classes,
            dropout=ccfg.dropout,
            pool=getattr(ccfg, "pool", "mean"),
            easy_idx=tuple(getattr(hcfg, "easy_idx", (0, 1, 2, 4))),
            hard_idx=tuple(getattr(hcfg, "hard_idx", (3, 5, 6, 7))),
        ).to(device)

    return EmotionClassifier(
        d_model=d_model,
        hidden_dim=ccfg.hidden_dim,
        n_classes=ccfg.n_classes,
        dropout=ccfg.dropout,
        pool=getattr(ccfg, "pool", "mean"),
    ).to(device)
