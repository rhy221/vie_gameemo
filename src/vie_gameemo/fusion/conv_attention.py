"""Conv-Attention 4-modality fusion (Section 7 of spec).

The recommended fusion module, adapted from Emotion-LLaMAv2 (2026) for our
4-modality setting (audio + face + context + text). Beats MULT, Q-Former,
and pure attention in the original paper's ablations.

Architecture:
    Input: 4 modality token sequences, each (B, T_i, 768).

    1. MLP per modality: standardize each to d_model channels.
    2. Align all sequence lengths to a target T via interpolation/broadcast.
    3. Form two hybrid structures:
        F_d = Concat all 4 along channel dim → (B, T, 4*768)
        F_s = Stack all 4 along last dim    → (B, T, 768, 4)
    4. Two parallel branches:
        ConvBranch: 4 residual Conv1d blocks with Switch activation on F_d
        AttentionBranch: MLP on F_d → softmax weights over 4 modalities
    5. Combine: u_fusion = F_conv + F_attn → (B, T, 768)
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vie_gameemo.fusion import register_fusion

logger = logging.getLogger(__name__)


class SwitchActivation(nn.Module):
    """Switch activation: x * sigmoid(x), similar to SiLU/Swish."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class _ResidualConvBlock(nn.Module):
    """One residual block: Conv1d + Switch + Conv1d + residual."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.act = SwitchActivation()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv2(self.act(self.conv1(x)))
        return out + residual


class ConvBranch(nn.Module):
    """Convolutional branch: capture local temporal patterns.

    Args:
        in_dim: Input channel dim (= n_modalities * d_model).
        hidden_dim: Output channel dim (= d_model).
        n_blocks: Number of residual conv blocks.
        kernel_size: Conv1d kernel size.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_blocks: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.initial_conv = nn.Conv1d(
            in_dim, hidden_dim, kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.blocks = nn.ModuleList([
            _ResidualConvBlock(hidden_dim, kernel_size) for _ in range(n_blocks)
        ])

    def forward(self, F_d: Tensor) -> Tensor:
        """Forward.

        Args:
            F_d: Concatenated features (B, T, in_dim).

        Returns:
            (B, T, hidden_dim) tensor.
        """
        x = F_d.transpose(1, 2)  # (B, in_dim, T)
        x = self.initial_conv(x)  # (B, hidden_dim, T)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)  # (B, T, hidden_dim)


class AttentionBranch(nn.Module):
    """Attention branch: per-timestep softmax weighting over modalities.

    Args:
        in_dim: = n_modalities * d_model.
        n_modalities: Number of modalities (4 for our pipeline).
    """

    def __init__(self, in_dim: int, n_modalities: int = 4) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.weight_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, n_modalities),
        )

    def forward(self, F_d: Tensor, F_s: Tensor) -> tuple[Tensor, Tensor]:
        """Forward.

        Args:
            F_d: Concatenated features (B, T, n_mods * d_model).
            F_s: Stacked features (B, T, d_model, n_mods).

        Returns:
            Tuple of:
                - F_attn: (B, T, d_model) weighted modality sum.
                - attn_weights: (B, T, n_mods) softmax weights for interpretability.
        """
        weights = self.weight_mlp(F_d)          # (B, T, n_mods)
        weights = F.softmax(weights, dim=-1)      # normalize over modalities
        # F_s: (B, T, D, n_mods); weights: (B, T, n_mods) → unsqueeze
        F_attn = (F_s * weights.unsqueeze(-2)).sum(dim=-1)  # (B, T, D)
        return F_attn, weights


@register_fusion("conv_attention_4m")
class ConvAttention4M(nn.Module):
    """Conv-Attention pre-fusion for 4 modalities (audio + face + context + text).

    Args:
        d_model: Per-modality hidden dim (768 for our encoders).
        n_modalities: 4 (audio, face, context, text).
        n_conv_blocks: Number of residual conv blocks in conv branch.
        kernel_size: Conv1d kernel size.
        align_to: Which modality's sequence length to use as target.
            One of: 'audio', 'face', 'context', 'text', or an int.
        return_attention: If True, forward returns (u_fusion, attn_weights).
        text_dim: Raw text-encoder output dim. Defaults to d_model. Set to
            1024 for XLM-R-large / CafeBERT, whose hidden size differs from the
            768-dim audio/visual encoders. The per-modality MLP projects it
            back to d_model so all modalities share the fused channel dim.
        audio_dim: Raw audio-encoder output dim (defaults to d_model).
        face_dim: Raw face-encoder output dim (defaults to d_model).
        context_dim: Raw context-encoder output dim (defaults to d_model).
    """

    def __init__(
        self,
        d_model: int = 768,
        n_modalities: int = 4,
        n_conv_blocks: int = 4,
        kernel_size: int = 3,
        align_to: str = "audio",
        return_attention: bool = True,
        text_dim: int | None = None,
        audio_dim: int | None = None,
        face_dim: int | None = None,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.align_to = align_to
        self.return_attention = return_attention

        # Per-modality MLP standardizers: project each encoder's native dim to d_model.
        # audio/face/context default to d_model (768); text may differ (e.g. 1024 for CafeBERT).
        self.mlp_audio   = nn.Linear(audio_dim   or d_model, d_model)
        self.mlp_face    = nn.Linear(face_dim    or d_model, d_model)
        self.mlp_context = nn.Linear(context_dim or d_model, d_model)
        self.mlp_text    = nn.Linear(text_dim    or d_model, d_model)

        in_dim = d_model * n_modalities
        self.conv_branch = ConvBranch(
            in_dim=in_dim,
            hidden_dim=d_model,
            n_blocks=n_conv_blocks,
            kernel_size=kernel_size,
        )
        self.attn_branch = AttentionBranch(
            in_dim=in_dim,
            n_modalities=n_modalities,
        )

    def forward(
        self,
        audio: Tensor,
        face: Tensor,
        context: Tensor,
        text: Tensor,
        has_face: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Fuse 4 modalities.

        Args:
            audio: (B, T_a, d_model).
            face: (B, T_f, d_model). Zeros if has_face=False.
            context: (B, T_c, d_model).
            text: (B, T_t, d_model).
            has_face: (B,) bool tensor. False entries get face zeroed.

        Returns:
            If return_attention: tuple (u_fusion, attn_weights).
            Else: u_fusion only.
            u_fusion: (B, T, d_model). attn_weights: (B, T, n_modalities).
        """
        # Zero out face for no-facecam clips
        if has_face is not None:
            mask = has_face.float().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            face = face * mask

        # MLP standardization
        u_a = self.mlp_audio(audio)
        u_f = self.mlp_face(face)
        u_c = self.mlp_context(context)
        u_t = self.mlp_text(text)

        # Determine target sequence length
        modality_map = {"audio": u_a, "face": u_f, "context": u_c, "text": u_t}
        if self.align_to in modality_map:
            target_len = modality_map[self.align_to].shape[1]
        else:
            target_len = int(self.align_to)

        # Align all sequences
        u_a = self._align_to_length(u_a, target_len)
        u_f = self._align_to_length(u_f, target_len)
        u_c = self._align_to_length(u_c, target_len)
        u_t = self._align_to_length(u_t, target_len)

        # Build hybrid structures
        F_d = torch.cat([u_a, u_f, u_c, u_t], dim=-1)          # (B, T, 4*D)
        F_s = torch.stack([u_a, u_f, u_c, u_t], dim=-1)         # (B, T, D, 4)

        # Two branches
        F_conv = self.conv_branch(F_d)                           # (B, T, D)
        F_attn, attn_weights = self.attn_branch(F_d, F_s)       # (B, T, D), (B, T, 4)

        u_fusion = F_conv + F_attn                               # (B, T, d_model)

        if self.return_attention:
            return u_fusion, attn_weights
        return u_fusion

    def _align_to_length(self, x: Tensor, target_len: int) -> Tensor:
        """Align sequence length to target_len via interpolation or broadcast.

        Args:
            x: (B, T, D).
            target_len: Desired sequence length.

        Returns:
            (B, target_len, D) tensor.
        """
        T = x.shape[1]
        if T == target_len:
            return x
        if T == 1:
            # Broadcast single token to target_len
            return x.expand(-1, target_len, -1)
        # Interpolate
        x_t = x.transpose(1, 2)  # (B, D, T)
        x_interp = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_interp.transpose(1, 2)  # (B, target_len, D)
