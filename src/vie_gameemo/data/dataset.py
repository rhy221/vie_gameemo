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
        label_schema: 'gaming_8' | 'ekman_7'.
        split_manifest: Optional path to a JSON mapping clip_id → split name.
            If None, a manifest is created from all annotations with a default split.
    """

    def __init__(
        self,
        annotations_dir: Path,
        features_dir: Path,
        split: str = "train",
        mode: str = "cached",
        label_schema: str = "gaming_8",
        split_manifest: Path | None = None,
        zero_modalities: list[str] | None = None,
        precomputed_augment_copies: int = 0,
    ) -> None:
        self.annotations_dir = Path(annotations_dir)
        self.features_dir = Path(features_dir)
        self.split = split
        self.mode = mode
        self.label_schema = label_schema
        # Modalities to zero at load time (strategy masking, not baked into cache).
        # e.g. ['context'] for face_only, [] for dual_path/full_frame.
        self.zero_modalities: frozenset[str] = frozenset(zero_modalities or [])
        # Precomputed raw-augmentation variants (see extract_features.py
        # --augment-copies): only meaningful for split="train" — val/test
        # must stay on the deterministic, unaugmented original for a stable
        # evaluation signal across epochs/runs.
        self.precomputed_augment_copies = precomputed_augment_copies if split == "train" else 0

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

            # Precomputed augmented variants for this clip (only those that
            # actually got cached — e.g. a clip with no frames wouldn't have
            # produced one, or --augment-copies K might differ run to run).
            variant_ids = [clip_id]
            if self.precomputed_augment_copies > 0:
                for i in range(self.precomputed_augment_copies):
                    if (self.features_dir / f"{clip_id}_aug{i}.pt").exists():
                        variant_ids.append(f"{clip_id}_aug{i}")

            try:
                ann = Annotation.load(ann_file)
                self.items.append({
                    "clip_id": clip_id,
                    "ann_path": ann_file,
                    "label": _LABEL_TO_IDX.get(ann.emotion_label.value, 0),
                    "has_face": ann.webcam_bbox is not None,
                    "reasoning_text": getattr(ann, "reasoning", ""),
                    "audio_desc": getattr(ann, "audio_tone_desc", ""),
                    "visual_desc": getattr(ann, "face_description", ""),
                    "transcript": getattr(ann, "transcript", ""),
                    "source_language": getattr(ann, "source_language", "vi"),
                    "variant_ids": variant_ids,
                })
            except Exception as exc:
                logger.warning("Failed to load annotation %s: %s", ann_file, exc)

        logger.info("Loaded %d samples for split=%s", len(self.items), split)
        if self.precomputed_augment_copies > 0:
            n_with_variants = sum(1 for it in self.items if len(it["variant_ids"]) > 1)
            logger.info(
                "Precomputed augment variants: %d/%d clips have >=1 cached variant "
                "(requested %d copies/clip)",
                n_with_variants, len(self.items), self.precomputed_augment_copies,
            )

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
        # Randomly pick among the original + any cached augmented variants
        # (see extract_features.py --augment-copies) each time this sample is
        # loaded — a fresh draw per call, so repeated epochs see different
        # variants rather than one fixed augmented copy per sample.
        variant_ids = item.get("variant_ids", [clip_id])
        load_id = random.choice(variant_ids) if len(variant_ids) > 1 else clip_id
        feat_path = self.features_dir / f"{load_id}.pt"
        features = torch.load(feat_path, map_location="cpu", weights_only=False)

        # Squeeze batch dim if present
        audio = _squeeze_batch(features.get("audio", torch.zeros(64, 768)))
        face = _squeeze_batch(features.get("face", torch.zeros(1, 768)))
        context = _squeeze_batch(features.get("context", torch.zeros(1, 768)))
        text = _squeeze_batch(features.get("text", torch.zeros(1, 768)))

        # Prefer the has_face resolved at extraction time (annotation.webcam_bbox
        # OR the webcam_bboxes.json fallback — see extract_features.py), since
        # that's what actually decided the face-crop strategy for this clip.
        # item["has_face"] (annotation-only) is a stale fallback for caches
        # extracted before this field existed. Getting this wrong silently
        # zeroes `face` for every affected clip via fusion's has_face masking,
        # independent of any explicit --zero-modality flag.
        cached_has_face = features.get("has_face")
        if cached_has_face is not None:
            has_face = bool(cached_has_face.item() if torch.is_tensor(cached_has_face) else cached_has_face)
        else:
            has_face = item["has_face"]

        # Strategy masking: zero modalities at load time (never baked into cache).
        if self.zero_modalities:
            if "audio" in self.zero_modalities:
                audio = torch.zeros_like(audio)
            if "face" in self.zero_modalities:
                face = torch.zeros_like(face)
                # has_face otherwise still reflects real webcam detection, so
                # fusion's attention-mask modality gating (see ConvAttention4M/
                # AttnOnly) would keep giving face a nonzero attention weight
                # on has_face=True samples even though its value was just
                # force-zeroed — weakening this explicit ablation for exactly
                # the samples where a real face existed.
                has_face = False
            if "context" in self.zero_modalities:
                context = torch.zeros_like(context)
            if "text" in self.zero_modalities:
                text = torch.zeros_like(text)

        return {
            "audio": audio,
            "face": face,
            "context": context,
            "text": text,
            "label": item["label"],
            "has_face": has_face,
            "clip_id": clip_id,
            "reasoning_text": item.get("reasoning_text", ""),
            "audio_desc": item.get("audio_desc", ""),
            "visual_desc": item.get("visual_desc", ""),
            "transcript": item.get("transcript", ""),
            "source_language": item.get("source_language", "vi"),
        }


def zero_modalities_for_strategy(strategy: str) -> list[str]:
    """Return which modalities should be zeroed for a given visual strategy.

    This is the single source of truth for strategy → modality masking.

    Strategy semantics:
      dual_path  — face crop + full-frame context, all 4 real.
      face_only  — no context encoder; context zeroed at load time.
      full_frame — both face and context use full-frame encoding (no webcam
                   crop); both tensors are real, nothing zeroed.
    """
    return {
        "face_only": ["context"],
        "dual_path": [],
        "full_frame": [],
    }.get(strategy, [])


def make_splits(
    annotations_dir: Path,
    split_ratios: dict[str, float] | None = None,
    seed: int = 42,
    stratify_by: list[str] | None = None,
    output_path: Path | None = None,
    method: str = "iterative_multilabel",
) -> dict[str, list[str]]:
    """Create stratified train/val/test splits over annotation IDs.

    Supports iterative multilabel stratification (trent-b) for joint
    stratification on (emotion, source_language, genre, codeswitch_bucket).

    Args:
        annotations_dir: Path to annotations.
        split_ratios: Dict like {'train': 0.7, 'val': 0.15, 'test_id': 0.1, 'test_ood': 0.05}.
        seed: RNG seed.
        stratify_by: Fields to stratify by.
        output_path: If set, save the manifest JSON here.
        method: "iterative_multilabel" | "random".

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

    if method == "iterative_multilabel" and stratify_by:
        assignments = _iterative_multilabel_split(
            ann_files, split_ratios, seed, stratify_by,
        )
    elif stratify_by:
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

    # Print distribution report
    _print_split_report(ann_files, assignments)

    return splits


def _iterative_multilabel_split(
    ann_files: list[Path],
    split_ratios: dict[str, float],
    seed: int,
    stratify_on: list[str],
) -> dict[str, str]:
    """Iterative multilabel stratification using trent-b's library.

    Encodes (emotion, source_language, genre, codeswitch_bucket) as a
    binary label matrix, then uses MultilabelStratifiedKFold to produce
    balanced splits.
    """
    import numpy as np

    annotations = []
    clip_ids = []
    for f in ann_files:
        try:
            ann = Annotation.load(f)
            annotations.append(ann)
            clip_ids.append(f.stem)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f, exc)

    if not annotations:
        return {}

    label_columns: dict[str, set[str]] = {}
    rows: list[dict[str, str]] = []
    for ann in annotations:
        row = {}
        if "emotion" in stratify_on:
            row["emotion"] = ann.emotion_label.value
            label_columns.setdefault("emotion", set()).add(ann.emotion_label.value)
        if "source_language" in stratify_on:
            lang = getattr(ann, "source_language", "vi")
            row["source_language"] = lang
            label_columns.setdefault("source_language", set()).add(lang)
        if "genre" in stratify_on:
            row["genre"] = ann.genre.value
            label_columns.setdefault("genre", set()).add(ann.genre.value)
        if "codeswitch_bucket" in stratify_on:
            ratio = getattr(ann, "code_switching_ratio", 0.0)
            bucket = "none" if ratio < 0.1 else ("low" if ratio < 0.3 else "high")
            row["codeswitch_bucket"] = bucket
            label_columns.setdefault("codeswitch_bucket", set()).add(bucket)
        rows.append(row)

    # Build binary label matrix
    all_labels = []
    for col_name, values in sorted(label_columns.items()):
        for val in sorted(values):
            all_labels.append((col_name, val))

    Y = np.zeros((len(rows), len(all_labels)), dtype=int)
    for i, row in enumerate(rows):
        for j, (col_name, val) in enumerate(all_labels):
            if row.get(col_name) == val:
                Y[i, j] = 1

    # Use MultilabelStratifiedKFold to create splits
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError:
        logger.warning(
            "iterative-stratification not installed. "
            "Run: pip install iterative-stratification. Falling back to random split."
        )
        rng = random.Random(seed)
        ids = list(clip_ids)
        rng.shuffle(ids)
        return _assign_splits(ids, split_ratios)

    split_names = list(split_ratios.keys())
    split_fracs = list(split_ratios.values())

    # Strategy: use k-fold where k = round(1/smallest_frac) to approximate ratios
    # First split: train vs rest, then split rest into val/test_id/test_ood
    n = len(clip_ids)
    assignments: dict[str, str] = {}

    train_frac = split_fracs[0]
    rest_frac = 1.0 - train_frac
    k_outer = max(2, round(1.0 / rest_frac))

    mskf = MultilabelStratifiedKFold(n_splits=k_outer, shuffle=True, random_state=seed)
    X_dummy = np.zeros((n, 1))

    train_idx, rest_idx = next(mskf.split(X_dummy, Y))

    for idx in train_idx:
        assignments[clip_ids[idx]] = split_names[0]

    if len(split_names) > 2:
        rest_clip_ids = [clip_ids[i] for i in rest_idx]
        rest_Y = Y[rest_idx]

        remaining_ratios = {k: v for k, v in zip(split_names[1:], split_fracs[1:])}
        total_remaining = sum(remaining_ratios.values())
        normalized = {k: v / total_remaining for k, v in remaining_ratios.items()}

        rest_rng = random.Random(seed + 1)
        rest_ids_shuffled = list(rest_clip_ids)
        rest_rng.shuffle(rest_ids_shuffled)
        sub_assignments = _assign_splits(rest_ids_shuffled, normalized)
        assignments.update(sub_assignments)
    elif len(split_names) == 2:
        for idx in rest_idx:
            assignments[clip_ids[idx]] = split_names[1]

    return assignments


def _print_split_report(
    ann_files: list[Path],
    assignments: dict[str, str],
) -> None:
    """Print emotion × source_language distribution per split."""
    from collections import Counter

    ann_cache: dict[str, Annotation] = {}
    for f in ann_files:
        try:
            ann_cache[f.stem] = Annotation.load(f)
        except Exception:
            pass

    splits = sorted(set(assignments.values()))
    emotions = [e.value for e in EmotionLabel]
    languages = sorted({getattr(ann_cache.get(cid), "source_language", "vi") for cid in assignments})

    logger.info("=" * 70)
    logger.info("SPLIT DISTRIBUTION REPORT: emotion × source_language")
    logger.info("=" * 70)

    for split in splits:
        split_ids = [cid for cid, s in assignments.items() if s == split]
        logger.info("\n--- %s (n=%d) ---", split, len(split_ids))

        counts: dict[str, Counter] = {e: Counter() for e in emotions}
        for cid in split_ids:
            ann = ann_cache.get(cid)
            if ann is None:
                continue
            lang = getattr(ann, "source_language", "vi")
            counts[ann.emotion_label.value][lang] += 1

        header = f"  {'emotion':<12}" + "".join(f"{l:>6}" for l in languages) + f"{'total':>8}"
        logger.info(header)
        logger.info("  " + "-" * len(header.strip()))
        for e in emotions:
            row = f"  {e:<12}"
            total = 0
            for l in languages:
                c = counts[e][l]
                row += f"{c:>6}"
                total += c
            row += f"{total:>8}"
            logger.info(row)

    logger.info("=" * 70)


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
