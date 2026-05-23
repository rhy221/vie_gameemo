"""Per-genre evaluation breakdown.

Splits test predictions by game genre (MOBA, FPS, Horror, Casual, RPG, Mobile)
and computes metrics for each. Used to demonstrate:
    - Genre-specific performance (some genres harder than others)
    - Strategy A (full-frame) suffers on genres with NPC face cutscenes
      (Horror, RPG) — see Section 5.1 of spec for rationale
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path

from vie_gameemo.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

_EMOTION_LABELS = ["neutral", "focus", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]
_LABEL2IDX = {l: i for i, l in enumerate(_EMOTION_LABELS)}


def per_genre_metrics(
    predictions: list[dict],
    genre_field: str = "genre",
) -> dict[str, dict]:
    """Compute metrics broken down by genre.

    Args:
        predictions: List of dicts with keys: 'y_true', 'y_pred', and genre_field.
            y_true / y_pred may be int indices or string label names.
        genre_field: Field name in predictions to group by.

    Returns:
        Dict mapping genre → metrics dict (same structure as compute_metrics).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for pred in predictions:
        genre = pred.get(genre_field, "unknown")
        groups[genre].append(pred)

    result: dict[str, dict] = {}
    for genre, preds in groups.items():
        y_true = [_to_idx(p["y_true"]) for p in preds]
        y_pred = [_to_idx(p["y_pred"]) for p in preds]
        n_classes = len(_EMOTION_LABELS)
        metrics = compute_metrics(y_true, y_pred, n_classes=n_classes, label_names=_EMOTION_LABELS)
        metrics.pop("confusion_matrix", None)
        metrics["n_samples"] = len(preds)
        result[genre] = metrics
        logger.info("Genre=%s  n=%d  macro_f1=%.4f  uar=%.4f",
                    genre, len(preds), metrics["macro_f1"], metrics["uar"])

    return result


def format_genre_table(genre_metrics: dict[str, dict], metric: str = "macro_f1") -> str:
    """Format per-genre results as Markdown table.

    Args:
        genre_metrics: Output of per_genre_metrics.
        metric: Which metric column to show prominently.

    Returns:
        Markdown table string.
    """
    if not genre_metrics:
        return "_No genre results_"

    cols = ["Genre", "N", "Accuracy", "Macro F1", "Weighted F1", "UAR"]
    rows = [f"| {' | '.join(cols)} |", f"| {' | '.join(['---'] * len(cols))} |"]
    for genre, m in sorted(genre_metrics.items()):
        row = "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            genre,
            m.get("n_samples", "?"),
            m.get("accuracy", 0),
            m.get("macro_f1", 0),
            m.get("weighted_f1", 0),
            m.get("uar", 0),
        )
        rows.append(row)
    return "\n".join(rows)


def save_genre_report(genre_metrics: dict[str, dict], output_path: Path) -> None:
    """Save per-genre report as CSV + Markdown side-by-side.

    Args:
        genre_metrics: Output of per_genre_metrics.
        output_path: Base path (e.g., 'outputs/results/per_genre.csv').
            A .md file is also written alongside the .csv.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["genre", "n_samples", "accuracy", "macro_f1", "weighted_f1", "uar"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for genre, m in sorted(genre_metrics.items()):
            writer.writerow({
                "genre": genre,
                "n_samples": m.get("n_samples", 0),
                "accuracy": round(m.get("accuracy", 0), 6),
                "macro_f1": round(m.get("macro_f1", 0), 6),
                "weighted_f1": round(m.get("weighted_f1", 0), 6),
                "uar": round(m.get("uar", 0), 6),
            })

    md_path = output_path.with_suffix(".md")
    md_path.write_text(format_genre_table(genre_metrics), encoding="utf-8")
    logger.info("Genre report saved → %s (+ .md)", output_path)


def _to_idx(label: str | int) -> int:
    if isinstance(label, int):
        return label
    return _LABEL2IDX.get(str(label).strip().lower(), 6)
