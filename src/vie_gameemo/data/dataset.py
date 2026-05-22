"""PyTorch Dataset for Vie-GameEmo.

Two modes:
    1. Raw mode: loads clip → runs encoders on-the-fly (slow, for inference)
    2. Cached mode: loads precomputed features from data/features/ (fast, for training)

Default and recommended is cached mode. Use `extract_features.py` script first
to populate the cache, then training reads from cache.
"""

import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vie_gameemo.data.schemas import Annotation, EmotionLabel

logger = logging.getLogger(__name__)

_LABEL_TO_IDX: dict[str, int] = {e.value: i for i, e in enumerate(EmotionLabel)}
_SPLIT_KEY = "split"


class VieGameEmoDataset(Dataset):
    """Vie-GameEmo PyTorch Dataset.

    Args:
        annotations_dir: Path to directory of Annotation JSON files.
        features_dir: Path to directory of cached features (.pt files).
        split: 'train' | 'val' | 'test_id' | 'test_ood'.
        mode: 'cached' (load .pt) | 'raw' (encode on-the-fly, needs encoders).
        label_schema: 'gaming_7' | 'ekman_7'.
        split_manifest: Optional path to a JSON mapping clip_id → split name.
            If None, a manifest is created from all annotations with a default split.
    """

    def __init__(
        self,
        annotations_dir: Path,
        features_dir: Path,
        split: str = "train",
        mode: str = "cached",
        label_schema: str = "gaming_7",
        split_manifest: Path | None = None,
    ) -> None:
        self.annotations_dir = Path(annotations_dir)
        self.features_dir = Path(features_dir)
        self.split = split
        self.mode = mode
        self.label_schema = label_schema

        annotation_files = sorted(self.annotations_dir.glob("*.json"))
        if not annotation_files:
            logger.warning("No annotation files found in %s", annotations_dir)

        # Load or create split manifest
        if split_manifest is not None and Path(split_manifest).exists():
            with open(split_manifest, encoding="utf-8") as f:
                clip_splits: dict[str, str] = json.load(f)
        else:
            clip_splits = _default_split_manifest(annotation_files)

        # Filter by current split
        self.items: list[dict] = []
        for ann_file in annotation_files:
            clip_id = ann_file.stem
            if clip_splits.get(clip_id) != split:
                continue

            if mode == "cached":
                feat_path = self.features_dir / f"{clip_id}.pt"
                if not feat_path.exists():
                    logger.debug("No cached features for %s; skipping", clip_id)
                    continue

            try:
                ann = Annotation.load(ann_file)
                self.items.append({
                    "clip_id": clip_id,
                    "ann_path": ann_file,
                    "label": _LABEL_TO_IDX.get(ann.emotion_label.value, 0),
                    "has_face": ann.webcam_bbox is not None,
                })
            except Exception as exc:
                logger.warning("Failed to load annotation %s: %s", ann_file, exc)

        logger.info("Loaded %d samples for split=%s", len(self.items), split)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample dict.

        Returns:
            Dict with keys:
                audio: (T_a, 768)
                face: (T_f, 768)
                context: (T_c, 768)
                text: (T_t, 768)
                label: int
                has_face: bool
                clip_id: str
        """
        item = self.items[idx]
        clip_id = item["clip_id"]

        if self.mode == "cached":
            return self._load_cached(item)
        else:
            raise NotImplementedError("Raw mode requires encoders; use cached mode for training")

    def _load_cached(self, item: dict) -> dict:
        """Load precomputed features from .pt file.

        Args:
            item: Item dict from self.items.

        Returns:
            Sample dict with modality tensors.
        """
        clip_id = item["clip_id"]
        feat_path = self.features_dir / f"{clip_id}.pt"
        features = torch.load(feat_path, map_location="cpu", weights_only=False)

        # Squeeze batch dim if present
        audio = _squeeze_batch(features.get("audio", torch.zeros(64, 768)))
        face = _squeeze_batch(features.get("face", torch.zeros(1, 768)))
        context = _squeeze_batch(features.get("context", torch.zeros(1, 768)))
        text = _squeeze_batch(features.get("text", torch.zeros(1, 768)))

        return {
            "audio": audio,
            "face": face,
            "context": context,
            "text": text,
            "label": item["label"],
            "has_face": item["has_face"],
            "clip_id": clip_id,
        }


def make_splits(
    annotations_dir: Path,
    split_ratios: dict[str, float] | None = None,
    seed: int = 42,
    stratify_by: list[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, list[str]]:
    """Create stratified train/val/test splits over annotation IDs.

    Args:
        annotations_dir: Path to annotations.
        split_ratios: Dict like {'train': 0.7, 'val': 0.15, 'test_id': 0.1, 'test_ood': 0.05}.
        seed: RNG seed.
        stratify_by: Fields to stratify by (currently 'emotion_label' supported).
        output_path: If set, save the manifest JSON here.

    Returns:
        Dict mapping split_name → list of clip_ids.
    """
    if split_ratios is None:
        split_ratios = {"train": 0.70, "val": 0.15, "test_id": 0.10, "test_ood": 0.05}

    ann_files = sorted(Path(annotations_dir).glob("*.json"))
    if not ann_files:
        return {k: [] for k in split_ratios}

    rng = random.Random(seed)
    clip_ids = [f.stem for f in ann_files]

    if stratify_by:
        # Group by label for stratification
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for f in ann_files:
            try:
                ann = Annotation.load(f)
                groups[ann.emotion_label.value].append(f.stem)
            except Exception:
                groups["unknown"].append(f.stem)

        assignments: dict[str, str] = {}
        for group_clips in groups.values():
            rng.shuffle(group_clips)
            assignments.update(_assign_splits(group_clips, split_ratios))
    else:
        rng.shuffle(clip_ids)
        assignments = _assign_splits(clip_ids, split_ratios)

    splits: dict[str, list[str]] = {k: [] for k in split_ratios}
    for cid, sname in assignments.items():
        splits[sname].append(cid)

    if output_path:
        Path(output_path).write_text(json.dumps(assignments, indent=2), encoding="utf-8")
        logger.info("Split manifest saved to %s", output_path)

    return splits


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch, padding sequences to max length within the batch.

    Args:
        batch: List of dicts from __getitem__.

    Returns:
        Batched dict with padded tensors and metadata.
    """
    clip_ids = [b["clip_id"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    has_face = torch.tensor([b["has_face"] for b in batch], dtype=torch.bool)

    audio = _pad_and_stack([b["audio"] for b in batch])
    face = _pad_and_stack([b["face"] for b in batch])
    context = _pad_and_stack([b["context"] for b in batch])
    text = _pad_and_stack([b["text"] for b in batch])

    return {
        "audio": audio,
        "face": face,
        "context": context,
        "text": text,
        "label": labels,
        "has_face": has_face,
        "clip_id": clip_ids,
    }


def _pad_and_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Pad variable-length (T, D) tensors to max T in batch, then stack.

    Args:
        tensors: List of (T_i, D) tensors.

    Returns:
        (B, T_max, D) tensor, zero-padded.
    """
    if not tensors:
        return torch.zeros(0, 1, 768)
    max_t = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[-1]
    padded = torch.zeros(len(tensors), max_t, D)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0]] = t
    return padded


def _squeeze_batch(t: torch.Tensor) -> torch.Tensor:
    """Remove leading batch dim if present (1, T, D) → (T, D)."""
    if t.dim() == 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def _default_split_manifest(ann_files: list[Path], seed: int = 42) -> dict[str, str]:
    """Create a default 70/15/10/5 split manifest.

    Args:
        ann_files: Annotation JSON files.
        seed: RNG seed.

    Returns:
        Dict mapping clip_id → split name.
    """
    clip_ids = [f.stem for f in ann_files]
    rng = random.Random(seed)
    rng.shuffle(clip_ids)
    ratios = {"train": 0.70, "val": 0.15, "test_id": 0.10, "test_ood": 0.05}
    return _assign_splits(clip_ids, ratios)


def _assign_splits(clip_ids: list[str], ratios: dict[str, float]) -> dict[str, str]:
    """Assign split names to clip_ids proportionally.

    Args:
        clip_ids: Ordered list of clip IDs.
        ratios: Split name → fraction.

    Returns:
        Dict mapping clip_id → split name.
    """
    n = len(clip_ids)
    splits = list(ratios.keys())
    fracs = list(ratios.values())
    boundaries = [0]
    for f in fracs:
        boundaries.append(boundaries[-1] + int(n * f))
    boundaries[-1] = n  # ensure all clips assigned

    assignments: dict[str, str] = {}
    for i, sname in enumerate(splits):
        for cid in clip_ids[boundaries[i]:boundaries[i + 1]]:
            assignments[cid] = sname
    return assignments
