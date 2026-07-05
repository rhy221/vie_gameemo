"""Feature caching: precompute frozen encoder outputs.

Since encoders (Whisper, ViT-FER, ViT-ImageNet, XLM-R) are frozen during
training, running them every batch wastes compute. This module precomputes
their outputs once and stores as .pt files.

Cache is invalidated when:
    - Underlying clip changes (hash mismatch)
    - Encoder config changes (model name, target tokens, etc.)
    - User explicitly requests recompute
"""

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_META_SUFFIX = ".meta.json"


def cache_features(
    clip_id: str,
    features: dict[str, torch.Tensor],
    cache_dir: Path,
    overwrite: bool = False,
    config_hash: str = "",
    clip_hash: str = "",
) -> Path:
    """Save precomputed features to cache.

    Args:
        clip_id: Unique clip identifier.
        features: Dict mapping modality name → tensor. Expected keys:
            'audio', 'face', 'context', 'text', and optionally 'has_face'.
        cache_dir: Cache directory.
        overwrite: If False and cache exists, skip.
        config_hash: Hash of encoder config (for invalidation).
        clip_hash: Hash of source clip file (for invalidation).

    Returns:
        Path to saved .pt file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pt_path = cache_dir / f"{clip_id}.pt"
    meta_path = cache_dir / f"{clip_id}{_META_SUFFIX}"

    if pt_path.exists() and not overwrite:
        logger.debug("Cache hit (skip): %s", clip_id)
        return pt_path

    cpu_features = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in features.items()}
    torch.save(cpu_features, pt_path)

    meta = {
        "clip_id": clip_id,
        "config_hash": config_hash,
        "clip_hash": clip_hash,
        "shapes": {k: list(v.shape) for k, v in features.items()
                   if isinstance(v, torch.Tensor)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.debug("Cached features: %s", clip_id)
    return pt_path


def infer_dim_from_cache(cache_dir: Path, modality: str) -> int | None:
    """Infer a modality's feature dim from one cached clip's `.meta.json`.

    Lets fusion pick up e.g. the audio dim automatically after switching
    `audio_encoder.type`/`model_name`, instead of requiring the matching
    `fusion.audio_dim` to be kept in sync by hand.

    Args:
        cache_dir: Feature cache directory (`cfg.paths.features`).
        modality: Modality key as stored in `cache_features` shapes, e.g.
            "audio", "face", "context", "text".

    Returns:
        Last-axis dim of the cached tensor, or None if no cache entry with
        that modality's shape recorded is found.
    """
    if not cache_dir.exists():
        return None
    for meta_path in cache_dir.glob(f"*{_META_SUFFIX}"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        shape = meta.get("shapes", {}).get(modality)
        if shape:
            return shape[-1]
    return None


def load_cached_features(
    clip_id: str,
    cache_dir: Path,
) -> dict[str, torch.Tensor] | None:
    """Load cached features.

    Args:
        clip_id: Unique clip identifier.
        cache_dir: Cache directory.

    Returns:
        Dict of tensors, or None if not cached.
    """
    pt_path = cache_dir / f"{clip_id}.pt"
    if not pt_path.exists():
        return None
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Failed to load cache for %s: %s", clip_id, exc)
        return None


def is_cache_valid(
    clip_id: str,
    clip_path: Path,
    cache_dir: Path,
    config_hash: str,
) -> bool:
    """Check whether cached features are still valid.

    Args:
        clip_id: Clip identifier.
        clip_path: Path to source clip.
        cache_dir: Cache directory.
        config_hash: Hash of relevant config (model names + token counts).

    Returns:
        True if cache exists and matches current config + clip hashes.
    """
    pt_path = cache_dir / f"{clip_id}.pt"
    meta_path = cache_dir / f"{clip_id}{_META_SUFFIX}"

    if not pt_path.exists() or not meta_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if meta.get("config_hash") != config_hash:
        return False

    if clip_path.exists():
        from vie_gameemo.utils.io import file_hash
        current_clip_hash = file_hash(clip_path)
        if meta.get("clip_hash") and meta["clip_hash"] != current_clip_hash:
            return False

    return True
