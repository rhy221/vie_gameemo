"""Tests for loss functions."""

import pytest
import torch

from vie_gameemo.training.losses import FocalLoss, make_class_weights

B = 4
N_CLASSES = 8


class TestFocalLoss:
    def test_output_is_scalar(self):
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_no_reduction(self):
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction="none")
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (B,)

    def test_sum_reduction(self):
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction="sum")
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gamma_zero_equals_ce(self):
        torch.manual_seed(42)
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))
        focal = FocalLoss(alpha=1.0, gamma=0.0)(logits, targets)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.allclose(focal, ce, atol=1e-5)

    def test_per_class_alpha(self):
        alpha = torch.ones(N_CLASSES)
        alpha[0] = 2.0
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        logits = torch.randn(B, N_CLASSES)
        targets = torch.zeros(B, dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flow(self):
        loss_fn = FocalLoss()
        logits = torch.randn(B, N_CLASSES, requires_grad=True)
        targets = torch.randint(0, N_CLASSES, (B,))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_loss_is_nonnegative(self):
        loss_fn = FocalLoss()
        logits = torch.randn(B, N_CLASSES)
        targets = torch.randint(0, N_CLASSES, (B,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0


class TestMakeClassWeights:
    def test_inverse_freq(self):
        labels = [0, 0, 0, 1, 1, 2]
        weights = make_class_weights(labels, n_classes=3, method="inverse_freq")
        assert weights.shape == (3,)
        assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-5)

    def test_effective_number(self):
        labels = [0, 0, 0, 1, 1, 2]
        weights = make_class_weights(labels, n_classes=3, method="effective_number")
        assert weights.shape == (3,)
        assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown class weight method"):
            make_class_weights([0, 1], n_classes=2, method="bogus")

    def test_rare_class_gets_higher_weight(self):
        labels = [0] * 100 + [1] * 10 + [2] * 1
        weights = make_class_weights(labels, n_classes=3, method="inverse_freq")
        assert weights[2] > weights[1] > weights[0]

    def test_empty_classes_handled(self):
        labels = [0, 0, 0]
        weights = make_class_weights(labels, n_classes=3, method="inverse_freq")
        assert weights.shape == (3,)
        assert not torch.isnan(weights).any()
