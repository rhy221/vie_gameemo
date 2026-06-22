"""Tests for attention visualization module."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import torch


class TestAttentionViz:
    def test_plot_timeline_numpy(self, tmp_path):
        from vie_gameemo.evaluation.attention_viz import plot_attention_timeline

        weights = np.random.dirichlet([1, 1, 1, 1], size=64)
        fig = plot_attention_timeline(weights, clip_id="test",
                                      save_path=tmp_path / "timeline.png")
        assert (tmp_path / "timeline.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_timeline_tensor(self, tmp_path):
        from vie_gameemo.evaluation.attention_viz import plot_attention_timeline

        weights = torch.randn(1, 64, 4).softmax(dim=-1)
        fig = plot_attention_timeline(weights, clip_id="test_tensor")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_stacked(self, tmp_path):
        from vie_gameemo.evaluation.attention_viz import plot_attention_stacked

        weights = np.random.dirichlet([1, 1, 1, 1], size=64)
        fig = plot_attention_stacked(weights, save_path=tmp_path / "stacked.png")
        assert (tmp_path / "stacked.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_by_genre(self, tmp_path):
        from vie_gameemo.evaluation.attention_viz import plot_mean_attention_by_genre

        preds = [
            {"genre": "moba", "attn_weights": np.random.dirichlet([1, 1, 1, 1], size=64)},
            {"genre": "moba", "attn_weights": np.random.dirichlet([1, 1, 1, 1], size=64)},
            {"genre": "fps", "attn_weights": np.random.dirichlet([1, 1, 1, 1], size=64)},
        ]
        fig = plot_mean_attention_by_genre(preds, save_path=tmp_path / "genre.png")
        assert (tmp_path / "genre.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_by_emotion(self, tmp_path):
        from vie_gameemo.evaluation.attention_viz import plot_mean_attention_by_emotion

        preds = [
            {"y_true": 0, "attn_weights": np.random.dirichlet([1, 1, 1, 1], size=64)},
            {"y_true": 1, "attn_weights": np.random.dirichlet([1, 1, 1, 1], size=64)},
            {"y_true": 2, "attn_weights": np.random.dirichlet([1, 1, 1, 1], size=64)},
        ]
        fig = plot_mean_attention_by_emotion(preds, save_path=tmp_path / "emotion.png")
        assert (tmp_path / "emotion.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty_predictions(self):
        from vie_gameemo.evaluation.attention_viz import plot_mean_attention_by_genre

        fig = plot_mean_attention_by_genre([])
        import matplotlib.pyplot as plt
        plt.close(fig)
