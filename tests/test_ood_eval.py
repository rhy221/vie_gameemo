"""Tests for OOD evaluation module."""

import json

import pytest

from vie_gameemo.evaluation.ood_eval import (
    format_ood_comparison,
    ood_id_comparison,
)


class TestOodIdComparison:
    def test_basic_comparison(self):
        id_m = {"accuracy": 0.85, "macro_f1": 0.80, "weighted_f1": 0.82, "uar": 0.78, "n_samples": 60}
        ood_m = {"accuracy": 0.70, "macro_f1": 0.65, "weighted_f1": 0.68, "uar": 0.63, "n_samples": 30}
        result = ood_id_comparison(id_m, ood_m)

        assert result["accuracy"]["in_distribution"] == 0.85
        assert result["accuracy"]["out_of_distribution"] == 0.70
        assert result["accuracy"]["delta"] == pytest.approx(-0.15, abs=1e-4)
        assert result["id_n_samples"] == 60
        assert result["ood_n_samples"] == 30

    def test_perfect_match(self):
        m = {"accuracy": 0.90, "macro_f1": 0.88, "weighted_f1": 0.89, "uar": 0.87, "n_samples": 50}
        result = ood_id_comparison(m, m)
        assert result["accuracy"]["delta"] == 0.0

    def test_ood_better_than_id(self):
        id_m = {"accuracy": 0.70, "macro_f1": 0.65, "weighted_f1": 0.68, "uar": 0.63, "n_samples": 60}
        ood_m = {"accuracy": 0.80, "macro_f1": 0.75, "weighted_f1": 0.78, "uar": 0.73, "n_samples": 30}
        result = ood_id_comparison(id_m, ood_m)
        assert result["accuracy"]["delta"] > 0


class TestFormatOodComparison:
    def test_format_produces_markdown(self):
        comparison = {
            "accuracy": {"in_distribution": 0.85, "out_of_distribution": 0.70, "delta": -0.15, "delta_pct": -17.65},
            "macro_f1": {"in_distribution": 0.80, "out_of_distribution": 0.65, "delta": -0.15, "delta_pct": -18.75},
            "weighted_f1": {"in_distribution": 0.82, "out_of_distribution": 0.68, "delta": -0.14, "delta_pct": -17.07},
            "uar": {"in_distribution": 0.78, "out_of_distribution": 0.63, "delta": -0.15, "delta_pct": -19.23},
            "id_n_samples": 60,
            "ood_n_samples": 30,
        }
        table = format_ood_comparison(comparison)
        assert "Metric" in table
        assert "In-Distribution" in table
        assert "accuracy" in table
        assert "|" in table
