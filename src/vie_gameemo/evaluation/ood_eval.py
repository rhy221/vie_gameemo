"""Out-of-distribution evaluation (R1-Omni-inspired, Section 12.2 of spec).

Two OOD evaluation protocols:
    1. Unseen streamer: leave-one-streamer-out — train on all streamers except
       one, test on the held-out streamer.
    2. Cross-genre: train on a subset of genres, test on unseen genres.

These reveal generalization capability beyond in-distribution accuracy.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from vie_gameemo.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

_EMOTION_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


def evaluate_ood_split(
    fusion: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: "DataLoader",
    device: torch.device,
    n_classes: int = 8,
) -> dict:
    """Evaluate fusion + classifier on a DataLoader (any split).

    Args:
        fusion: Fusion module.
        classifier: Classifier module.
        loader: DataLoader for the OOD split.
        device: Torch device.
        n_classes: Number of emotion classes.

    Returns:
        Metrics dict (accuracy, macro_f1, weighted_f1, uar, per_class_f1).
    """
    fusion.eval()
    classifier.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            context = batch["context"].to(device)
            text = batch["text"].to(device)
            labels = batch["label"].to(device)
            has_face = batch.get("has_face")
            if has_face is not None:
                has_face = has_face.to(device)

            fused = fusion(audio, face, context, text, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]
            logits = classifier(fused)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, n_classes=n_classes,
                              label_names=_EMOTION_LABELS)
    metrics.pop("confusion_matrix", None)
    metrics["n_samples"] = len(all_labels)
    return metrics


def ood_id_comparison(
    id_metrics: dict,
    ood_metrics: dict,
) -> dict:
    """Compare in-distribution vs OOD metrics.

    Args:
        id_metrics: Metrics on in-distribution test split.
        ood_metrics: Metrics on OOD test split.

    Returns:
        Dict with id, ood, and delta for each metric.
    """
    comparison = {}
    for key in ("accuracy", "macro_f1", "weighted_f1", "uar"):
        id_val = id_metrics.get(key, 0.0)
        ood_val = ood_metrics.get(key, 0.0)
        comparison[key] = {
            "in_distribution": round(id_val, 4),
            "out_of_distribution": round(ood_val, 4),
            "delta": round(ood_val - id_val, 4),
            "delta_pct": round((ood_val - id_val) / max(id_val, 1e-8) * 100, 2),
        }
    comparison["id_n_samples"] = id_metrics.get("n_samples", 0)
    comparison["ood_n_samples"] = ood_metrics.get("n_samples", 0)
    return comparison


def leave_one_streamer_out(
    annotations_dir: Path,
    features_dir: Path,
) -> dict[str, list[str]]:
    """Group clips by streamer for leave-one-out evaluation.

    Args:
        annotations_dir: Directory of annotation JSON files.
        features_dir: Directory of cached features.

    Returns:
        Dict mapping streamer name → list of clip IDs.
    """
    streamer_clips: dict[str, list[str]] = defaultdict(list)

    for ann_path in sorted(Path(annotations_dir).glob("*.json")):
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        clip_id = ann_path.stem
        feat_path = Path(features_dir) / f"{clip_id}.pt"
        if not feat_path.exists():
            continue

        streamer = data.get("streamer", "unknown")
        streamer_clips[streamer].append(clip_id)

    logger.info("Found %d streamers: %s",
                len(streamer_clips),
                {k: len(v) for k, v in streamer_clips.items()})
    return dict(streamer_clips)


def cross_genre_splits(
    annotations_dir: Path,
    features_dir: Path,
    train_genres: list[str],
    test_genres: list[str],
) -> tuple[list[str], list[str]]:
    """Split clips by genre for cross-genre evaluation.

    Args:
        annotations_dir: Directory of annotation JSON files.
        features_dir: Directory of cached features.
        train_genres: Genres to include in training.
        test_genres: Genres to include in testing.

    Returns:
        Tuple of (train_clip_ids, test_clip_ids).
    """
    train_clips, test_clips = [], []

    for ann_path in sorted(Path(annotations_dir).glob("*.json")):
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        clip_id = ann_path.stem
        feat_path = Path(features_dir) / f"{clip_id}.pt"
        if not feat_path.exists():
            continue

        genre = data.get("genre", "unknown")
        if genre in train_genres:
            train_clips.append(clip_id)
        elif genre in test_genres:
            test_clips.append(clip_id)

    logger.info("Cross-genre split: train=%d clips (%s), test=%d clips (%s)",
                len(train_clips), train_genres, len(test_clips), test_genres)
    return train_clips, test_clips


def format_ood_comparison(comparison: dict) -> str:
    """Format OOD vs ID comparison as Markdown table.

    Args:
        comparison: Output of ood_id_comparison.

    Returns:
        Markdown table string.
    """
    cols = ["Metric", "In-Distribution", "Out-of-Distribution", "Delta", "Delta %"]
    rows = [f"| {' | '.join(cols)} |", f"| {' | '.join(['---'] * len(cols))} |"]

    for key in ("accuracy", "macro_f1", "weighted_f1", "uar"):
        if key not in comparison:
            continue
        m = comparison[key]
        rows.append(
            "| {} | {:.4f} | {:.4f} | {:+.4f} | {:+.2f}% |".format(
                key, m["in_distribution"], m["out_of_distribution"],
                m["delta"], m["delta_pct"],
            )
        )

    rows.append(f"| N samples | {comparison.get('id_n_samples', '?')} "
                f"| {comparison.get('ood_n_samples', '?')} | — | — |")
    return "\n".join(rows)


def save_ood_report(
    comparison: dict,
    output_path: Path,
) -> None:
    """Save OOD comparison report as JSON + Markdown.

    Args:
        comparison: Output of ood_id_comparison.
        output_path: Base path (e.g., 'outputs/results/ood_report.json').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_path = output_path.with_suffix(".md")
    md_path.write_text(format_ood_comparison(comparison), encoding="utf-8")
    logger.info("OOD report saved → %s (+ .md)", output_path)
