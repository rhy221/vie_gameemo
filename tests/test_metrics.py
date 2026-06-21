"""Tests for evaluation metrics."""

import numpy as np
import pytest

from vie_gameemo.evaluation.metrics import (
    cohens_kappa,
    compute_metrics,
    format_confusion_matrix,
)

LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 3, 4, 5, 6, 7]
        y_pred = [0, 1, 2, 3, 4, 5, 6, 7]
        m = compute_metrics(y_true, y_pred, n_classes=8, label_names=LABELS)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["weighted_f1"] == 1.0
        assert m["uar"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 1, 2, 3]
        y_pred = [1, 2, 3, 0]
        m = compute_metrics(y_true, y_pred, n_classes=8, label_names=LABELS)
        assert m["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 2, 3, 4, 5, 6, 7]
        y_pred = [0, 1, 2, 3, 4, 5, 6, 7]
        m = compute_metrics(y_true, y_pred, n_classes=8)
        assert m["confusion_matrix"].shape == (8, 8)

    def test_per_class_f1_keys(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        m = compute_metrics(y_true, y_pred, n_classes=8, label_names=LABELS)
        for label in LABELS:
            assert label in m["per_class_f1"]

    def test_numpy_input(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        m = compute_metrics(y_true, y_pred, n_classes=4)
        assert m["accuracy"] == 1.0

    def test_single_class(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        m = compute_metrics(y_true, y_pred, n_classes=8)
        assert m["accuracy"] == 1.0

    def test_uar_vs_accuracy(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 0]
        m = compute_metrics(y_true, y_pred, n_classes=8)
        assert m["accuracy"] == 0.8
        assert m["uar"] < m["accuracy"]


class TestFormatConfusionMatrix:
    def test_basic_format(self):
        cm = np.array([[5, 1], [2, 4]])
        result = format_confusion_matrix(cm, label_names=["a", "b"])
        assert "a" in result
        assert "b" in result
        assert "5" in result

    def test_no_labels(self):
        cm = np.eye(3, dtype=int)
        result = format_confusion_matrix(cm)
        assert "0" in result
        assert "1" in result


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = [0, 1, 2, 3, 4]
        b = [0, 1, 2, 3, 4]
        assert cohens_kappa(a, b) == 1.0

    def test_no_agreement(self):
        a = [0, 0, 0, 0]
        b = [1, 1, 1, 1]
        kappa = cohens_kappa(a, b)
        assert kappa <= 0.0

    def test_partial_agreement(self):
        a = [0, 1, 2, 3, 4]
        b = [0, 1, 2, 0, 0]
        kappa = cohens_kappa(a, b)
        assert 0.0 < kappa < 1.0
