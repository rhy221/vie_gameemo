"""Tests for Stage 4 MLP emotion classifier."""

import pytest
import torch

from vie_gameemo.classifiers.mlp import EmotionClassifier

B = 2
D = 768
T = 64
N_CLASSES = 8


class TestEmotionClassifier:
    def test_output_shape_3d_input(self):
        clf = EmotionClassifier(d_model=D, hidden_dim=256, n_classes=N_CLASSES)
        x = torch.randn(B, T, D)
        logits = clf(x)
        assert logits.shape == (B, N_CLASSES)

    def test_output_shape_2d_input(self):
        clf = EmotionClassifier(d_model=D, hidden_dim=256, n_classes=N_CLASSES)
        x = torch.randn(B, D)
        logits = clf(x)
        assert logits.shape == (B, N_CLASSES)

    @pytest.mark.parametrize("pool", ["mean", "max", "cls", "attention"])
    def test_pool_modes(self, pool):
        clf = EmotionClassifier(d_model=D, hidden_dim=256, n_classes=N_CLASSES, pool=pool)
        x = torch.randn(B, T, D)
        logits = clf(x)
        assert logits.shape == (B, N_CLASSES)

    def test_unknown_pool_raises(self):
        clf = EmotionClassifier(d_model=D, hidden_dim=256, n_classes=N_CLASSES, pool="invalid")
        x = torch.randn(B, T, D)
        with pytest.raises(ValueError, match="Unknown pool"):
            clf(x)

    def test_gradient_flow(self):
        clf = EmotionClassifier(d_model=D, hidden_dim=256, n_classes=N_CLASSES)
        x = torch.randn(B, T, D, requires_grad=True)
        logits = clf(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_n_classes(self):
        for nc in [2, 7, 8, 10]:
            clf = EmotionClassifier(d_model=D, hidden_dim=128, n_classes=nc)
            logits = clf(torch.randn(B, T, D))
            assert logits.shape == (B, nc)

    def test_dropout_in_eval(self):
        clf = EmotionClassifier(d_model=D, hidden_dim=256, n_classes=N_CLASSES, dropout=0.5)
        clf.eval()
        x = torch.randn(1, T, D)
        out1 = clf(x)
        out2 = clf(x)
        assert torch.allclose(out1, out2)
