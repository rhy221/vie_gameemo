"""Classification metrics: accuracy, F1 variants, UAR, confusion matrix."""

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    n_classes: int,
    label_names: list[str] | None = None,
) -> dict:
    """Compute full metrics dict.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_classes: Number of classes.
        label_names: Optional class names for confusion matrix readability.

    Returns:
        Dict with keys:
            - accuracy: float
            - macro_f1: float
            - weighted_f1: float
            - uar: float (unweighted average recall)
            - per_class_f1: dict[class_name → float]
            - confusion_matrix: 2D ndarray
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    all_classes = list(range(n_classes))

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=all_classes, average="weighted", zero_division=0))
    uar = float(recall_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0))

    per_class_f1_vals = f1_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
    per_class_recall_vals = recall_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
    per_class_precision_vals = precision_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
    if label_names is None:
        label_names = [str(i) for i in range(n_classes)]
    n = min(len(label_names), len(per_class_f1_vals))
    per_class_f1 = {label_names[i]: float(per_class_f1_vals[i]) for i in range(n)}
    per_class_recall = {label_names[i]: float(per_class_recall_vals[i]) for i in range(n)}
    per_class_precision = {label_names[i]: float(per_class_precision_vals[i]) for i in range(n)}

    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "uar": uar,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "confusion_matrix": cm,
    }


def format_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str] | None = None,
) -> str:
    """Pretty-print confusion matrix as aligned text table.

    Args:
        cm: Confusion matrix array (n_classes × n_classes).
        label_names: Class names.

    Returns:
        Formatted string table.
    """
    n = cm.shape[0]
    names = label_names if label_names else [str(i) for i in range(n)]
    col_w = max(len(n) for n in names) + 2
    header = " " * col_w + "".join(n.rjust(col_w) for n in names)
    rows = [header]
    for i, row_name in enumerate(names):
        row = row_name.rjust(col_w) + "".join(str(cm[i, j]).rjust(col_w) for j in range(n))
        rows.append(row)
    return "\n".join(rows)


def cohens_kappa(
    annotator_a: list[int],
    annotator_b: list[int],
) -> float:
    """Cohen's kappa for inter-annotator agreement (used in pilot Stage 0).

    Args:
        annotator_a: Labels from annotator A.
        annotator_b: Labels from annotator B.

    Returns:
        Cohen's kappa coefficient.
    """
    return float(cohen_kappa_score(annotator_a, annotator_b))
