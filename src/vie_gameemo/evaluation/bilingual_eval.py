"""Bilingual evaluation: per-language metrics + fragmentation diagnostic.

Implements:
  B — F1 split by test-VI and test-EN subsets
  C — Per-class F1 for rare classes across ablation variants
  D — Modality ablation breakdown by language
  E — Fragmentation diagnostic: t-SNE/UMAP of h_text colored by language
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def evaluate_per_language(
    preds: list[int],
    labels: list[int],
    languages: list[str],
    n_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """Compute metrics separately for VI and EN test subsets (Ablation B).

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.
        languages: Per-sample source language ("vi"/"en").
        n_classes: Number of classes.
        class_names: Optional label names.

    Returns:
        Dict with 'vi' and 'en' sub-dicts, each containing macro_f1, uar,
        per_class_f1, per_class_recall.
    """
    from vie_gameemo.training.losses import per_class_metrics

    results = {}
    for lang in ("vi", "en"):
        mask = [i for i, l in enumerate(languages) if l == lang]
        if not mask:
            results[lang] = {"n": 0, "macro_f1": 0.0, "uar": 0.0}
            continue

        lang_preds = [preds[i] for i in mask]
        lang_labels = [labels[i] for i in mask]
        metrics = per_class_metrics(lang_preds, lang_labels, n_classes, class_names)

        from sklearn.metrics import recall_score
        uar = recall_score(lang_labels, lang_preds, average="macro", zero_division=0)

        results[lang] = {
            "n": len(mask),
            "macro_f1": metrics["macro_f1"],
            "uar": float(uar),
            "per_class_f1": metrics["per_class_f1"],
            "per_class_recall": metrics["per_class_recall"],
        }

    return results


def rare_class_report(
    per_language_results: dict,
    rare_classes: list[str] | None = None,
) -> dict:
    """Extract per-class F1 for rare classes across languages (Ablation C).

    Args:
        per_language_results: Output of evaluate_per_language.
        rare_classes: Class names considered rare. Default: disgusted, fear, shocked.

    Returns:
        Dict[class_name][language] = F1 score.
    """
    if rare_classes is None:
        rare_classes = ["disgusted", "fear", "shocked"]

    report = {}
    for cls in rare_classes:
        report[cls] = {}
        for lang in ("vi", "en"):
            lang_data = per_language_results.get(lang, {})
            f1_dict = lang_data.get("per_class_f1", {})
            report[cls][lang] = f1_dict.get(cls, 0.0)

    return report


def fragmentation_diagnostic(
    embeddings: np.ndarray | torch.Tensor,
    languages: list[str],
    output_dir: Path,
    method: str = "tsne",
    stage: str = "after_finetune",
) -> dict:
    """Visualize h_text embeddings colored by language (Ablation E).

    Produces t-SNE/UMAP plot + quantitative metrics:
    - Silhouette score (language as cluster label)
    - Language classifier accuracy from frozen h_text

    Args:
        embeddings: (N, D) text embeddings.
        languages: Per-sample language labels.
        output_dir: Where to save plots.
        method: "tsne" or "umap".
        stage: Label for the plot (e.g., "before_finetune", "after_finetune",
               "after_adversarial").

    Returns:
        Dict with 'silhouette', 'lang_classifier_acc', 'plot_path'.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_labels = np.array([0 if l == "vi" else 1 for l in languages])

    # Dimensionality reduction
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            logger.warning("umap-learn not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))

    coords_2d = reducer.fit_transform(embeddings)

    # Silhouette score
    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(embeddings, lang_labels)) if len(set(lang_labels)) > 1 else 0.0

    # Language classifier accuracy (logistic regression on frozen embeddings)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=500, random_state=42)
    if len(set(lang_labels)) > 1 and len(embeddings) > 10:
        cv_folds = min(5, min(np.bincount(lang_labels)))
        cv_folds = max(2, cv_folds)
        scores = cross_val_score(clf, embeddings, lang_labels, cv=cv_folds, scoring="accuracy")
        lang_acc = float(scores.mean())
    else:
        lang_acc = 0.5

    # Plot
    plot_path = output_dir / f"fragmentation_{stage}_{method}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2196F3" if l == 0 else "#FF5722" for l in lang_labels]
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, alpha=0.6, s=15)
        ax.set_title(f"h_text {method.upper()} — {stage}\n"
                     f"silhouette={sil:.3f}, lang_clf_acc={lang_acc:.3f}")
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", label="VI"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", label="EN"),
            ],
            loc="upper right",
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("Fragmentation plot saved: %s", plot_path)
    except Exception as exc:
        logger.warning("Could not save fragmentation plot: %s", exc)
        plot_path = None

    return {
        "silhouette": sil,
        "lang_classifier_acc": lang_acc,
        "plot_path": str(plot_path) if plot_path else None,
    }


def log_bilingual_report(
    per_lang: dict,
    rare_report: dict,
    frag: dict | None = None,
) -> None:
    """Log a summary of bilingual evaluation results."""
    logger.info("=" * 60)
    logger.info("BILINGUAL EVALUATION REPORT")
    logger.info("=" * 60)

    for lang in ("vi", "en"):
        data = per_lang.get(lang, {})
        logger.info(
            "  %s (n=%d): macro_F1=%.4f, UAR=%.4f",
            lang.upper(), data.get("n", 0), data.get("macro_f1", 0), data.get("uar", 0),
        )

    logger.info("\nRare class F1 by language:")
    for cls, lang_f1 in rare_report.items():
        logger.info("  %s: VI=%.4f, EN=%.4f", cls, lang_f1.get("vi", 0), lang_f1.get("en", 0))

    if frag:
        logger.info(
            "\nFragmentation: silhouette=%.4f, lang_clf_acc=%.4f",
            frag.get("silhouette", 0), frag.get("lang_classifier_acc", 0),
        )
    logger.info("=" * 60)
