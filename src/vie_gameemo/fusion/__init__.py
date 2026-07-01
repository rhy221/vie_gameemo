"""Stage 3 fusion modules.

Pre-fusion combines per-modality token sequences into a single fused
representation before the classifier and LLM.

Available fusion types (registered via `get_fusion`):
    - 'late'             — late fusion (concatenate logits, ablation baseline)
    - 'early'            — concat embeddings + MLP (ablation baseline)
    - 'mult'             — cross-modal Transformer (Tsai et al., ACL 2019)
    - 'q_former'         — Q-Former style (AffectGPT)
    - 'conv_only'        — only convolutional branch
    - 'attn_only'        — only attention branch
    - 'conv_attention_4m' — RECOMMENDED: Conv-Attention 4-modality (from Emotion-LLaMAv2)

The recommended fusion is `conv_attention_4m` (Section 7 of spec).
"""

from typing import Callable

from torch import nn

# Module registry. Populated by submodules via `register_fusion`.
_FUSION_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_fusion(name: str):
    """Decorator to register a fusion module under a name.

    Example:
        @register_fusion("conv_attention_4m")
        class ConvAttention4M(nn.Module): ...
    """
    def decorator(cls):
        _FUSION_REGISTRY[name] = cls
        return cls
    return decorator


def get_fusion(name: str, **kwargs) -> nn.Module:
    """Factory: instantiate a fusion module by name.

    Args:
        name: Registered fusion name (see module docstring).
        **kwargs: Passed to the module constructor.

    Returns:
        Instantiated nn.Module.

    Raises:
        KeyError: If name not registered.
    """
    # Ensure all fusion modules are imported (triggers @register_fusion)
    _ensure_registered()

    if name not in _FUSION_REGISTRY:
        raise KeyError(
            f"Unknown fusion '{name}'. Available: {sorted(_FUSION_REGISTRY)}"
        )
    return _FUSION_REGISTRY[name](**kwargs)


def modality_dim_kwargs(fcfg) -> dict[str, int]:
    """Collect per-modality encoder dim overrides set on the fusion config.

    Returns only the keys explicitly set (non-None) so they can be splatted
    into `get_fusion(...)` without breaking baseline fusions that don't accept
    them. Used when an encoder's native output dim differs from `d_model`
    (e.g. text_dim=1024 for XLM-R-large/CafeBERT vs 768 audio/visual).
    """
    kwargs: dict[str, int] = {}
    for key in ("text_dim", "audio_dim", "face_dim", "context_dim"):
        val = getattr(fcfg, key, None)
        if val is not None:
            kwargs[key] = val
    return kwargs


def _ensure_registered() -> None:
    """Import fusion modules if not yet registered."""
    if "late" not in _FUSION_REGISTRY:
        from vie_gameemo.fusion import baselines       # noqa: F401
    if "conv_attention_4m" not in _FUSION_REGISTRY:
        from vie_gameemo.fusion import conv_attention  # noqa: F401
