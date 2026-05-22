"""Strategy A vs B vs C ablation (Section 5.6 of spec).

This is one of the KEY contributions of the project. Compares 3 visual
encoding strategies on the same train/test split:

    Strategy A — Full-frame (paper baseline): no webcam detection, ViT-FER
                 directly on full frames.
    Strategy B — Face-only: webcam detect + crop, no context path.
    Strategy C — Dual-path (recommended): face crop + context full-frame.

For each strategy:
    1. Train fusion + classifier from the same Stage 1 setup
    2. Eval on same test split
    3. Per-genre breakdown (expect Strategy A to suffer on Horror / RPG)
"""

import copy
import csv
import logging
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_STRATEGY_CONFIGS: dict[str, dict] = {
    "A": {
        "description": "Full-frame baseline (no webcam detection)",
        "face_source": "full_frame",
        "use_context": False,
    },
    "B": {
        "description": "Face-only (webcam detect + crop, no context)",
        "face_source": "webcam_crop",
        "use_context": False,
    },
    "C": {
        "description": "Dual-path (face crop + context full-frame)",
        "face_source": "webcam_crop",
        "use_context": True,
    },
}


def run_strategy_ablation(
    cfg: SimpleNamespace,
    output_dir: Path,
    strategies: list[str] | None = None,
) -> dict:
    """Run A/B/C ablation, return aggregated results.

    For each strategy, trains a fresh perception model from cached features,
    then evaluates on the test_id split. Per-genre breakdown is included if
    the annotation data has genre labels.

    Args:
        cfg: Base config namespace.
        output_dir: Where to save per-strategy checkpoints and results.
        strategies: List of strategies to run. Default: ['A', 'B', 'C'].

    Returns:
        Dict mapping strategy → {'metrics': dict, 'genre_metrics': dict,
                                  'description': str, 'checkpoint': str}.
    """
    import torch
    from torch.utils.data import DataLoader

    from vie_gameemo.data.dataset import VieGameEmoDataset, collate_fn, make_splits
    from vie_gameemo.evaluation.metrics import compute_metrics
    from vie_gameemo.evaluation.per_genre import per_genre_metrics
    from vie_gameemo.training.perception import evaluate, train_perception

    strategies = strategies or ["A", "B", "C"]
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[str, dict] = {}

    for strategy_key in strategies:
        if strategy_key not in _STRATEGY_CONFIGS:
            logger.warning("Unknown strategy %s, skipping", strategy_key)
            continue

        strategy_info = _STRATEGY_CONFIGS[strategy_key]
        logger.info("--- Strategy %s: %s ---", strategy_key, strategy_info["description"])

        strategy_cfg = _patch_cfg_for_strategy(cfg, strategy_info)
        strategy_dir = output_dir / f"strategy_{strategy_key}"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        strategy_cfg.paths.checkpoints = str(strategy_dir)

        annotations_dir = Path(cfg.paths.annotations)
        splits_path = annotations_dir / "splits.json"
        if not splits_path.exists():
            make_splits(
                annotations_dir=annotations_dir,
                split_ratios=(0.70, 0.15, 0.10, 0.05),
                seed=cfg.seed,
                output_path=splits_path,
            )

        label_names = ["hype", "tilted", "focused", "disappointed", "shocked", "amused", "neutral"]
        label2idx = {l: i for i, l in enumerate(label_names)}

        train_ds = VieGameEmoDataset(
            annotations_dir=annotations_dir,
            features_dir=Path(cfg.paths.features),
            split="train",
            splits_path=splits_path,
            label2idx=label2idx,
        )
        val_ds = VieGameEmoDataset(
            annotations_dir=annotations_dir,
            features_dir=Path(cfg.paths.features),
            split="val",
            splits_path=splits_path,
            label2idx=label2idx,
        )
        test_ds = VieGameEmoDataset(
            annotations_dir=annotations_dir,
            features_dir=Path(cfg.paths.features),
            split="test_id",
            splits_path=splits_path,
            label2idx=label2idx,
        )

        pcfg = strategy_cfg.training.perception
        train_loader = DataLoader(
            train_ds, batch_size=pcfg.batch_size, shuffle=True,
            num_workers=getattr(pcfg, "num_workers", 0), collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=pcfg.batch_size, shuffle=False,
            num_workers=getattr(pcfg, "num_workers", 0), collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=pcfg.batch_size, shuffle=False,
            num_workers=getattr(pcfg, "num_workers", 0), collate_fn=collate_fn,
        )

        best_ckpt = train_perception(
            cfg=strategy_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        from vie_gameemo.classifiers.mlp import EmotionClassifier
        from vie_gameemo.fusion import get_fusion
        from vie_gameemo.training.perception import load_checkpoint

        fcfg = strategy_cfg.fusion
        ccfg = strategy_cfg.classifier
        fusion = get_fusion(
            fcfg.type,
            d_model=fcfg.d_model,
            n_modalities=fcfg.n_modalities,
            n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
            kernel_size=getattr(fcfg, "kernel_size", 3),
            align_to=getattr(fcfg, "align_to", "audio"),
            return_attention=False,
        ).to(device)
        classifier = EmotionClassifier(
            d_model=fcfg.d_model,
            hidden_dim=ccfg.hidden_dim,
            n_classes=ccfg.n_classes,
            dropout=ccfg.dropout,
        ).to(device)
        load_checkpoint(best_ckpt, fusion, classifier)

        metrics = evaluate(fusion=fusion, classifier=classifier,
                           loader=test_loader, device=device, n_classes=ccfg.n_classes)

        all_preds = _collect_predictions(fusion, classifier, test_loader, test_ds, device)
        genre_metrics = per_genre_metrics(all_preds, genre_field="genre")

        logger.info("Strategy %s | macro_f1=%.4f | uar=%.4f",
                    strategy_key, metrics["macro_f1"], metrics["uar"])

        results[strategy_key] = {
            "description": strategy_info["description"],
            "metrics": {k: float(v) for k, v in metrics.items() if k != "confusion_matrix"},
            "genre_metrics": genre_metrics,
            "checkpoint": str(best_ckpt),
        }

    _save_ablation_results(results, output_dir)
    return results


def format_ablation_table(
    results: dict,
    metric: str = "macro_f1",
) -> str:
    """Format ablation results as a Markdown table for the report.

    Args:
        results: Output of run_strategy_ablation.
        metric: Which metric to highlight.

    Returns:
        Markdown table string.
    """
    if not results:
        return "_No ablation results_"

    cols = ["Strategy", "Description", "Accuracy", "Macro F1", "Weighted F1", "UAR"]
    rows = [f"| {' | '.join(cols)} |", f"| {' | '.join(['---'] * len(cols))} |"]
    for key in sorted(results.keys()):
        r = results[key]
        m = r.get("metrics", {})
        row = "| {} | {} | {:.4f} | **{:.4f}** | {:.4f} | {:.4f} |".format(
            key,
            r.get("description", ""),
            m.get("accuracy", 0),
            m.get("macro_f1", 0),
            m.get("weighted_f1", 0),
            m.get("uar", 0),
        ) if metric == "macro_f1" else "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            key,
            r.get("description", ""),
            m.get("accuracy", 0),
            m.get("macro_f1", 0),
            m.get("weighted_f1", 0),
            m.get("uar", 0),
        )
        rows.append(row)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_cfg_for_strategy(cfg: SimpleNamespace, strategy_info: dict) -> SimpleNamespace:
    """Return a shallow copy of cfg with strategy-specific overrides."""
    import copy as _copy
    patched = _copy.deepcopy(cfg)
    patched.data.face_source = strategy_info["face_source"]
    patched.data.use_context = strategy_info["use_context"]
    return patched


def _collect_predictions(fusion, classifier, loader, dataset, device) -> list[dict]:
    """Run inference and return list of {y_true, y_pred, genre, clip_id}."""
    import torch

    fusion.eval()
    classifier.eval()
    preds = []
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
            pred_labels = logits.argmax(dim=-1)

            for i in range(len(labels)):
                clip_id = batch.get("clip_id", [""] * len(labels))
                genre = batch.get("genre", ["unknown"] * len(labels))
                preds.append({
                    "y_true": int(labels[i].item()),
                    "y_pred": int(pred_labels[i].item()),
                    "clip_id": clip_id[i] if isinstance(clip_id, (list, tuple)) else "",
                    "genre": genre[i] if isinstance(genre, (list, tuple)) else "unknown",
                })
    return preds


def _save_ablation_results(results: dict, output_dir: Path) -> None:
    """Save ablation comparison as CSV + Markdown."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ablation_results.json"
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            "description": v["description"],
            "metrics": v["metrics"],
            "checkpoint": v["checkpoint"],
        }
    json_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = output_dir / "ablation_table.md"
    md_path.write_text(format_ablation_table(results), encoding="utf-8")

    csv_path = output_dir / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["strategy", "description", "accuracy", "macro_f1", "weighted_f1", "uar"]
        )
        writer.writeheader()
        for key, r in sorted(results.items()):
            m = r.get("metrics", {})
            writer.writerow({
                "strategy": key,
                "description": r.get("description", ""),
                "accuracy": round(m.get("accuracy", 0), 6),
                "macro_f1": round(m.get("macro_f1", 0), 6),
                "weighted_f1": round(m.get("weighted_f1", 0), 6),
                "uar": round(m.get("uar", 0), 6),
            })
    logger.info("Ablation results saved → %s", output_dir)
