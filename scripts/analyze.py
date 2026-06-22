"""Post-training analysis: read eval results + annotations, produce actionable report.

Runs after eval.py. Reads the eval JSON + annotation files to produce:
  1. Per-class performance breakdown (F1, recall, precision, support)
  2. Per-language performance gap (VI vs EN)
  3. Top confusion pairs (which classes get mixed up)
  4. Rare class deep-dive (disgusted, fear, shocked)
  5. Fragmentation diagnostic (optional, needs features)
  6. Actionable recommendations

Usage:
    python scripts/analyze.py --config config.yaml
    python scripts/analyze.py --config config.yaml --eval-json outputs/results/eval.json
    python scripts/analyze.py --config config.yaml --with-fragmentation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import setup_logging

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]
RARE_CLASSES = ["disgusted", "fear", "shocked"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training analysis + recommendations")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--eval-json", type=Path, default=Path("outputs/results/eval.json"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint to run eval on-the-fly if eval.json missing")
    parser.add_argument("--with-fragmentation", action="store_true",
                        help="Run t-SNE fragmentation diagnostic (needs cached text features)")
    parser.add_argument("--output", type=Path, default=Path("outputs/results/analysis.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))

    # Load or run eval
    if args.eval_json.exists():
        with open(args.eval_json, encoding="utf-8") as f:
            eval_data = json.load(f)
        logger.info("Loaded eval results from %s", args.eval_json)
    elif args.checkpoint:
        logger.info("eval.json not found, running eval...")
        eval_data = _run_eval_inline(cfg, args.checkpoint)
    else:
        logger.error("No eval.json and no --checkpoint. Run eval.py first or provide --checkpoint.")
        return 1

    # Load annotations for language info
    annotations_dir = Path(cfg.paths.annotations)
    ann_map = _load_annotations(annotations_dir)

    report = {}

    # 1. Per-class breakdown
    metrics = eval_data.get("metrics", {})
    per_class = _per_class_breakdown(metrics)
    report["per_class"] = per_class
    _print_per_class(per_class)

    # 2. Confusion analysis
    cm = metrics.get("confusion_matrix")
    if cm:
        confusion = _confusion_analysis(np.array(cm))
        report["top_confusions"] = confusion
        _print_confusions(confusion)

    # 3. Per-language analysis
    lang_report = _per_language_analysis(eval_data, ann_map)
    if lang_report:
        report["per_language"] = lang_report
        _print_per_language(lang_report)

    # 4. Rare class deep-dive
    rare = _rare_class_analysis(per_class, cm, ann_map)
    report["rare_classes"] = rare
    _print_rare_classes(rare)

    # 5. Fragmentation diagnostic
    if args.with_fragmentation:
        frag = _fragmentation_diagnostic(cfg, ann_map)
        if frag:
            report["fragmentation"] = frag
            _print_fragmentation(frag)

    # 6. Recommendations
    recs = _generate_recommendations(report, metrics)
    report["recommendations"] = recs
    _print_recommendations(recs)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("\nFull report saved → %s", args.output)

    return 0


# ---------------------------------------------------------------------------
# 1. Per-class breakdown
# ---------------------------------------------------------------------------

def _per_class_breakdown(metrics: dict) -> list[dict]:
    per_f1 = metrics.get("per_class_f1", {})
    per_recall = metrics.get("per_class_recall", {})
    per_precision = metrics.get("per_class_precision", {})
    cm = metrics.get("confusion_matrix")
    support = {}
    if cm:
        cm_arr = np.array(cm)
        for i, label in enumerate(EMOTION_LABELS):
            if i < cm_arr.shape[0]:
                support[label] = int(cm_arr[i].sum())

    rows = []
    for label in EMOTION_LABELS:
        rows.append({
            "class": label,
            "f1": per_f1.get(label, 0.0),
            "recall": per_recall.get(label, 0.0),
            "precision": per_precision.get(label, 0.0),
            "support": support.get(label, 0),
        })
    rows.sort(key=lambda r: r["f1"])
    return rows


def _print_per_class(rows: list[dict]) -> None:
    logger.info("")
    logger.info("=" * 65)
    logger.info("PER-CLASS PERFORMANCE (sorted by F1, worst first)")
    logger.info("=" * 65)
    logger.info(f"  {'class':<12} {'F1':>6} {'Recall':>8} {'Prec':>8} {'Support':>9}")
    logger.info(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*9}")
    for r in rows:
        flag = " ⚠" if r["f1"] < 0.3 else ""
        logger.info(f"  {r['class']:<12} {r['f1']:>6.3f} {r['recall']:>8.3f} {r['precision']:>8.3f} {r['support']:>9}{flag}")


# ---------------------------------------------------------------------------
# 2. Confusion analysis
# ---------------------------------------------------------------------------

def _confusion_analysis(cm: np.ndarray, top_k: int = 5) -> list[dict]:
    n = min(cm.shape[0], len(EMOTION_LABELS))
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                row_total = cm[i].sum()
                pairs.append({
                    "true": EMOTION_LABELS[i],
                    "predicted": EMOTION_LABELS[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / row_total) if row_total > 0 else 0,
                })
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_k]


def _print_confusions(pairs: list[dict]) -> None:
    logger.info("")
    logger.info("TOP CONFUSION PAIRS")
    logger.info("-" * 50)
    for p in pairs:
        logger.info(f"  {p['true']:<12} → {p['predicted']:<12} : {p['count']:>4} ({p['rate']:.1%})")


# ---------------------------------------------------------------------------
# 3. Per-language analysis
# ---------------------------------------------------------------------------

def _per_language_analysis(eval_data: dict, ann_map: dict) -> dict | None:
    metrics = eval_data.get("metrics", {})
    cm = metrics.get("confusion_matrix")
    if not cm or not ann_map:
        return None

    lang_counts = Counter(a.get("source_language", "vi") for a in ann_map.values())
    if len(lang_counts) < 2:
        return None

    # Need per-sample predictions to split by language
    # If not available in eval_data, return just distribution
    per_lang = eval_data.get("per_language")
    if per_lang:
        return per_lang

    return {
        "distribution": dict(lang_counts),
        "note": "Per-language metrics require re-running eval with language-aware splits. "
                "Use bilingual_eval.evaluate_per_language() for detailed breakdown.",
    }


def _print_per_language(report: dict) -> None:
    logger.info("")
    logger.info("LANGUAGE DISTRIBUTION")
    logger.info("-" * 40)
    dist = report.get("distribution", {})
    for lang, count in sorted(dist.items()):
        logger.info(f"  {lang.upper()}: {count} clips")

    for lang in ("vi", "en"):
        data = report.get(lang)
        if data and isinstance(data, dict) and "macro_f1" in data:
            logger.info(f"  {lang.upper()} macro_F1={data['macro_f1']:.4f} UAR={data.get('uar', 0):.4f} (n={data.get('n', '?')})")

    note = report.get("note")
    if note:
        logger.info(f"  Note: {note}")


# ---------------------------------------------------------------------------
# 4. Rare class deep-dive
# ---------------------------------------------------------------------------

def _rare_class_analysis(per_class: list[dict], cm, ann_map: dict) -> dict:
    result = {}
    for label in RARE_CLASSES:
        row = next((r for r in per_class if r["class"] == label), None)
        if not row:
            continue

        # Count per language for this class
        lang_dist = Counter()
        for a in ann_map.values():
            if a.get("emotion_label") == label:
                lang_dist[a.get("source_language", "vi")] += 1

        # Top confusions for this class
        confusions = []
        if cm is not None:
            cm_arr = np.array(cm)
            idx = EMOTION_LABELS.index(label) if label in EMOTION_LABELS else -1
            if 0 <= idx < cm_arr.shape[0]:
                for j in range(cm_arr.shape[1]):
                    if j != idx and cm_arr[idx, j] > 0:
                        confusions.append({
                            "predicted_as": EMOTION_LABELS[j] if j < len(EMOTION_LABELS) else str(j),
                            "count": int(cm_arr[idx, j]),
                        })
                confusions.sort(key=lambda c: c["count"], reverse=True)

        result[label] = {
            "f1": row["f1"],
            "recall": row["recall"],
            "support": row["support"],
            "lang_distribution": dict(lang_dist),
            "top_confusions": confusions[:3],
        }
    return result


def _print_rare_classes(rare: dict) -> None:
    logger.info("")
    logger.info("RARE CLASS DEEP-DIVE")
    logger.info("-" * 50)
    for label, data in rare.items():
        logger.info(f"  {label}:")
        logger.info(f"    F1={data['f1']:.3f}  Recall={data['recall']:.3f}  Support={data['support']}")
        logger.info(f"    Language: {data['lang_distribution']}")
        if data["top_confusions"]:
            confused = ", ".join(f"{c['predicted_as']}({c['count']})" for c in data["top_confusions"])
            logger.info(f"    Most confused with: {confused}")


# ---------------------------------------------------------------------------
# 5. Fragmentation diagnostic
# ---------------------------------------------------------------------------

def _fragmentation_diagnostic(cfg, ann_map: dict) -> dict | None:
    features_dir = Path(cfg.paths.features)
    if not features_dir.exists():
        logger.warning("Features dir not found, skipping fragmentation diagnostic")
        return None

    import torch
    embeddings = []
    languages = []

    for clip_id, ann in ann_map.items():
        text_feat_path = features_dir / f"{clip_id}_text.pt"
        if not text_feat_path.exists():
            # Try combined feature file
            combined = features_dir / f"{clip_id}.pt"
            if combined.exists():
                data = torch.load(combined, map_location="cpu", weights_only=False)
                if "text" in data:
                    emb = data["text"]
                    if emb.dim() == 2:
                        emb = emb.mean(dim=0)
                    elif emb.dim() == 3:
                        emb = emb.squeeze(0).mean(dim=0)
                    embeddings.append(emb.numpy())
                    languages.append(ann.get("source_language", "vi"))
            continue

        feat = torch.load(text_feat_path, map_location="cpu", weights_only=False)
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        if feat.dim() == 2:
            feat = feat.mean(dim=0)
        embeddings.append(feat.numpy())
        languages.append(ann.get("source_language", "vi"))

    if len(embeddings) < 20:
        logger.warning("Not enough text features (%d) for fragmentation diagnostic", len(embeddings))
        return None

    emb_array = np.stack(embeddings)

    from vie_gameemo.evaluation.bilingual_eval import fragmentation_diagnostic
    output_dir = Path(cfg.paths.results) if hasattr(cfg.paths, "results") else Path("outputs/results")
    result = fragmentation_diagnostic(emb_array, languages, output_dir, stage="post_training")
    return result


def _print_fragmentation(frag: dict) -> None:
    logger.info("")
    logger.info("FRAGMENTATION DIAGNOSTIC")
    logger.info("-" * 40)
    logger.info(f"  Silhouette (language): {frag.get('silhouette', 0):.4f}")
    logger.info(f"  Language classifier accuracy: {frag.get('lang_classifier_acc', 0):.4f}")
    if frag.get("plot_path"):
        logger.info(f"  Plot: {frag['plot_path']}")
    if frag.get("silhouette", 0) > 0.2:
        logger.info("  ⚠ High silhouette → text embeddings cluster by language → consider enabling adversarial head")
    else:
        logger.info("  ✓ Low silhouette → embeddings are reasonably language-mixed")


# ---------------------------------------------------------------------------
# 6. Recommendations
# ---------------------------------------------------------------------------

def _generate_recommendations(report: dict, metrics: dict) -> list[str]:
    recs = []
    macro_f1 = metrics.get("macro_f1", 0)
    per_class = report.get("per_class", [])
    rare = report.get("rare_classes", {})
    confusions = report.get("top_confusions", [])
    frag = report.get("fragmentation")
    per_lang = report.get("per_language")

    # Overall performance
    if macro_f1 < 0.3:
        recs.append("CRITICAL: Macro F1 < 0.3 — model barely learning. Check data pipeline, features, label distribution.")
    elif macro_f1 < 0.5:
        recs.append("Macro F1 < 0.5 — consider: more data, longer training, or unfreezing encoder layers.")

    # Weak classes
    weak = [r for r in per_class if r["f1"] < 0.2 and r["support"] > 0]
    if weak:
        names = ", ".join(r["class"] for r in weak)
        recs.append(f"Classes with F1 < 0.2: [{names}] — need more training samples or augmentation (fused_mixup).")

    # Zero-recall classes
    zero_recall = [r for r in per_class if r["recall"] == 0 and r["support"] > 0]
    if zero_recall:
        names = ", ".join(r["class"] for r in zero_recall)
        recs.append(f"Classes with ZERO recall: [{names}] — model never predicts these. Increase class_weights or use balanced_batch sampler.")

    # Rare class specifics
    for label in RARE_CLASSES:
        data = rare.get(label, {})
        if data.get("f1", 1) < 0.15 and data.get("support", 0) > 0:
            recs.append(f"Rare class '{label}' (F1={data['f1']:.3f}, n={data['support']}): "
                        f"consider adding more EN clips for this class, or enable fused_mixup augmentation.")

    # Confusion pairs
    if confusions:
        top = confusions[0]
        if top["rate"] > 0.3:
            recs.append(f"High confusion: {top['true']} → {top['predicted']} ({top['rate']:.0%}). "
                        f"Review annotation quality for these two classes or add discriminative features.")

    # Language gap
    if per_lang and isinstance(per_lang, dict):
        vi_f1 = per_lang.get("vi", {}).get("macro_f1")
        en_f1 = per_lang.get("en", {}).get("macro_f1")
        if vi_f1 is not None and en_f1 is not None:
            gap = vi_f1 - en_f1
            if gap < -0.05:
                recs.append(f"VI underperforms EN by {abs(gap):.3f} macro_F1 — enable language_adversarial head (lambda_grl=0.1).")
            elif gap > 0.1:
                recs.append(f"EN underperforms VI by {gap:.3f} macro_F1 — EN data may be too noisy or domain-shifted.")

    # Fragmentation
    if frag:
        sil = frag.get("silhouette", 0)
        lang_acc = frag.get("lang_classifier_acc", 0.5)
        if sil > 0.2 or lang_acc > 0.8:
            recs.append(f"Text embeddings fragment by language (silhouette={sil:.3f}, lang_clf_acc={lang_acc:.3f}). "
                        f"Enable adversarial head: text_encoder.language_adversarial.enabled=true")

    # Class weights
    if per_class:
        support_vals = [r["support"] for r in per_class if r["support"] > 0]
        if support_vals:
            imbalance_ratio = max(support_vals) / max(1, min(support_vals))
            if imbalance_ratio > 10:
                recs.append(f"Class imbalance ratio {imbalance_ratio:.0f}:1 — ensure class_weights='inverse_freq' is enabled in config.")

    if not recs:
        recs.append("No critical issues detected. Consider fine-tuning hyperparameters (lr, epochs, dropout) for incremental gains.")

    return recs


def _print_recommendations(recs: list[str]) -> None:
    logger.info("")
    logger.info("=" * 65)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 65)
    for i, rec in enumerate(recs, 1):
        logger.info(f"  {i}. {rec}")
    logger.info("=" * 65)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_annotations(annotations_dir: Path) -> dict[str, dict]:
    ann_map = {}
    for p in sorted(annotations_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ann_map[p.stem] = data
        except Exception:
            pass
    return ann_map


def _run_eval_inline(cfg, checkpoint: Path) -> dict:
    """Run eval.py logic inline to get metrics."""
    import subprocess
    eval_json = Path("outputs/results/eval.json")
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(Path(__file__).parent / "eval.py"),
        "--config", str(cfg._config_path) if hasattr(cfg, "_config_path") else "config.yaml",
        "--checkpoint", str(checkpoint),
        "--output", str(eval_json),
    ], check=True)
    with open(eval_json, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    sys.exit(main())
