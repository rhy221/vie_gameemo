"""Tests for Stage 3 fusion modules."""

import pytest
import torch

from vie_gameemo.fusion import get_fusion, _ensure_registered, _FUSION_REGISTRY
from vie_gameemo.fusion.conv_attention import (
    AttentionBranch,
    ConvAttention4M,
    ConvBranch,
    SwitchActivation,
)

B = 2
D = 768
T = 64


class TestSwitchActivation:
    def test_output_shape(self):
        act = SwitchActivation()
        x = torch.randn(B, T, D)
        out = act(x)
        assert out.shape == (B, T, D)

    def test_zero_input(self):
        act = SwitchActivation()
        x = torch.zeros(B, T, D)
        out = act(x)
        assert torch.allclose(out, torch.zeros_like(out))


class TestConvBranch:
    def test_output_shape(self):
        branch = ConvBranch(in_dim=D * 4, hidden_dim=D, n_blocks=2)
        x = torch.randn(B, T, D * 4)
        out = branch(x)
        assert out.shape == (B, T, D)

    def test_single_block(self):
        branch = ConvBranch(in_dim=D * 4, hidden_dim=D, n_blocks=1)
        x = torch.randn(B, T, D * 4)
        out = branch(x)
        assert out.shape == (B, T, D)


class TestAttentionBranch:
    def test_output_shape(self):
        branch = AttentionBranch(in_dim=D * 4, n_modalities=4)
        F_d = torch.randn(B, T, D * 4)
        F_s = torch.randn(B, T, D, 4)
        F_attn, weights = branch(F_d, F_s)
        assert F_attn.shape == (B, T, D)
        assert weights.shape == (B, T, 4)

    def test_weights_sum_to_one(self):
        branch = AttentionBranch(in_dim=D * 4, n_modalities=4)
        F_d = torch.randn(B, T, D * 4)
        F_s = torch.randn(B, T, D, 4)
        _, weights = branch(F_d, F_s)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestConvAttention4M:
    def test_output_with_attention(self, batch_tensors):
        model = ConvAttention4M(d_model=D, n_conv_blocks=2, return_attention=True)
        out = model(
            batch_tensors["audio"],
            batch_tensors["face"],
            batch_tensors["context"],
            batch_tensors["text"],
        )
        assert isinstance(out, tuple)
        u_fusion, attn_weights = out
        assert u_fusion.shape == (B, 64, D)
        assert attn_weights.shape == (B, 64, 4)

    def test_output_without_attention(self, batch_tensors):
        model = ConvAttention4M(d_model=D, n_conv_blocks=2, return_attention=False)
        out = model(
            batch_tensors["audio"],
            batch_tensors["face"],
            batch_tensors["context"],
            batch_tensors["text"],
        )
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, 64, D)

    def test_no_face_masking(self, batch_tensors_no_face):
        model = ConvAttention4M(d_model=D, n_conv_blocks=2, return_attention=True)
        u_fusion, _ = model(
            batch_tensors_no_face["audio"],
            batch_tensors_no_face["face"],
            batch_tensors_no_face["context"],
            batch_tensors_no_face["text"],
            has_face=batch_tensors_no_face["has_face"],
        )
        assert u_fusion.shape == (B, 64, D)

    def test_sequence_alignment(self):
        model = ConvAttention4M(d_model=D, n_conv_blocks=1, align_to="audio")
        audio = torch.randn(B, 32, D)
        face = torch.randn(B, 8, D)
        context = torch.randn(B, 1, D)
        text = torch.randn(B, 5, D)
        u_fusion, _ = model(audio, face, context, text)
        assert u_fusion.shape[1] == 32

    def test_gradient_flow(self, batch_tensors):
        model = ConvAttention4M(d_model=D, n_conv_blocks=2)
        u_fusion, _ = model(
            batch_tensors["audio"],
            batch_tensors["face"],
            batch_tensors["context"],
            batch_tensors["text"],
        )
        loss = u_fusion.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestFusionRegistry:
    def test_ensure_registered(self):
        _ensure_registered()
        assert "conv_attention_4m" in _FUSION_REGISTRY
        assert "late" in _FUSION_REGISTRY
        assert "early" in _FUSION_REGISTRY
        assert "mult" in _FUSION_REGISTRY
        assert "q_former" in _FUSION_REGISTRY
        assert "conv_only" in _FUSION_REGISTRY
        assert "attn_only" in _FUSION_REGISTRY

    def test_get_fusion_conv_attention(self):
        model = get_fusion("conv_attention_4m", d_model=D, n_conv_blocks=1)
        assert isinstance(model, ConvAttention4M)

    def test_get_fusion_unknown_raises(self):
        with pytest.raises(KeyError):
            get_fusion("nonexistent_fusion_type")

    @pytest.mark.parametrize("fusion_type", [
        "late", "early", "mult", "q_former", "conv_only", "attn_only", "conv_attention_4m",
    ])
    def test_all_baselines_forward(self, fusion_type, batch_tensors):
        kwargs = {"d_model": D}
        if fusion_type in ("conv_only", "conv_attention_4m"):
            kwargs["n_conv_blocks"] = 1
        model = get_fusion(fusion_type, **kwargs)
        out = model(
            batch_tensors["audio"],
            batch_tensors["face"],
            batch_tensors["context"],
            batch_tensors["text"],
        )
        if isinstance(out, tuple):
            out = out[0]
        assert out.ndim in (2, 3)
        assert out.shape[0] == B
