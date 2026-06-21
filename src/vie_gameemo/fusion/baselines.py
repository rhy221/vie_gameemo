"""Baseline fusion modules for ablation comparison.

Implements: late, early, MULT, Q-Former, conv_only, attn_only.
All register via `@register_fusion(name)`.

Use these in ablation experiments to show that Conv-Attention 4M wins.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vie_gameemo.fusion import register_fusion


def _pool_sequence(x: Tensor) -> Tensor:
    """Mean-pool (B, T, D) → (B, D)."""
    return x.mean(dim=1)


@register_fusion("late")
class LateFusion(nn.Module):
    """Late fusion: per-modality mean-pooled representations averaged.

    Returns (B, T=1, D) where T=1 to maintain interface consistency with
    the Conv-Attention module. The classifier will pool this further.
    """

    def __init__(self, d_model: int = 768, n_classes: int = 8) -> None:
        super().__init__()
        self.proj_audio = nn.Linear(d_model, d_model)
        self.proj_face = nn.Linear(d_model, d_model)
        self.proj_context = nn.Linear(d_model, d_model)
        self.proj_text = nn.Linear(d_model, d_model)

    def forward(
        self,
        audio: Tensor,
        face: Tensor,
        context: Tensor,
        text: Tensor,
        has_face: Tensor | None = None,
    ) -> Tensor:
        """Fuse by averaging per-modality projected mean-pools.

        Returns:
            (B, 1, D) fused representation.
        """
        if has_face is not None:
            mask = has_face.float().view(-1, 1, 1)
            face = face * mask

        a = self.proj_audio(_pool_sequence(audio))
        f = self.proj_face(_pool_sequence(face))
        c = self.proj_context(_pool_sequence(context))
        t = self.proj_text(_pool_sequence(text))

        fused = (a + f + c + t) / 4.0  # (B, D)
        return fused.unsqueeze(1)       # (B, 1, D)


@register_fusion("early")
class EarlyFusion(nn.Module):
    """Early fusion: concat mean-pooled modalities and project."""

    def __init__(self, d_model: int = 768, n_modalities: int = 4) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * n_modalities, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        audio: Tensor,
        face: Tensor,
        context: Tensor,
        text: Tensor,
        has_face: Tensor | None = None,
    ) -> Tensor:
        """Concatenate mean-pooled modalities and project.

        Returns:
            (B, 1, D) fused representation.
        """
        if has_face is not None:
            mask = has_face.float().view(-1, 1, 1)
            face = face * mask

        a = _pool_sequence(audio)
        f = _pool_sequence(face)
        c = _pool_sequence(context)
        t = _pool_sequence(text)

        concat = torch.cat([a, f, c, t], dim=-1)  # (B, 4*D)
        fused = self.proj(concat)                   # (B, D)
        return fused.unsqueeze(1)                   # (B, 1, D)


@register_fusion("mult")
class MULTFusion(nn.Module):
    """Cross-modal Transformer fusion (MULT).

    Simplified: each modality cross-attends to a reference modality (audio),
    then all cross-attended representations are mean-pooled and projected.

    Reference: Tsai et al., ACL 2019.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 2,
        n_modalities: int = 4,
    ) -> None:
        super().__init__()
        # Each modality cross-attends to audio
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_modalities)
        ])
        self.proj = nn.Linear(d_model * n_modalities, d_model)

    def forward(
        self,
        audio: Tensor,
        face: Tensor,
        context: Tensor,
        text: Tensor,
        has_face: Tensor | None = None,
    ) -> Tensor:
        """Cross-modal fusion with audio as key/value.

        Returns:
            (B, T_audio, D) fused representation.
        """
        if has_face is not None:
            mask = has_face.float().view(-1, 1, 1)
            face = face * mask

        modalities = [audio, face, context, text]
        cross_outputs = []
        for i, (attn, mod) in enumerate(zip(self.cross_attn, modalities)):
            # Q from modality, K/V from audio
            out, _ = attn(mod, audio, audio)
            # Align to audio length for concatenation
            if out.shape[1] != audio.shape[1]:
                out = F.interpolate(out.transpose(1, 2), size=audio.shape[1],
                                    mode="linear", align_corners=False).transpose(1, 2)
            cross_outputs.append(out)

        concat = torch.cat(cross_outputs, dim=-1)  # (B, T_audio, 4*D)
        return self.proj(concat)                    # (B, T_audio, D)


@register_fusion("q_former")
class QFormerFusion(nn.Module):
    """Q-Former style fusion (AffectGPT).

    Learnable query tokens cross-attend to all modality tokens.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_queries: int = 32,
        n_heads: int = 8,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])

    def forward(
        self,
        audio: Tensor,
        face: Tensor,
        context: Tensor,
        text: Tensor,
        has_face: Tensor | None = None,
    ) -> Tensor:
        """Q-Former cross-attention over all modality tokens.

        Returns:
            (B, n_queries, D) fused representation.
        """
        if has_face is not None:
            mask = has_face.float().view(-1, 1, 1)
            face = face * mask

        B = audio.shape[0]
        # Concatenate all modalities as KV
        kv = torch.cat([audio, face, context, text], dim=1)  # (B, T_total, D)

        queries = self.queries.expand(B, -1, -1)  # (B, n_queries, D)
        for i, (ca, ffn) in enumerate(zip(self.cross_attn_layers, self.ffn_layers)):
            norm1 = self.norms[i * 2]
            norm2 = self.norms[i * 2 + 1]
            attn_out, _ = ca(norm1(queries), kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(norm2(queries))

        return queries  # (B, n_queries, D)


@register_fusion("conv_only")
class ConvOnly(nn.Module):
    """Ablation: only the conv branch of Conv-Attention 4M."""

    def __init__(
        self,
        d_model: int = 768,
        n_modalities: int = 4,
        n_conv_blocks: int = 4,
        kernel_size: int = 3,
        align_to: str = "audio",
    ) -> None:
        super().__init__()
        self.align_to = align_to
        self.d_model = d_model

        self.mlp_audio = nn.Linear(d_model, d_model)
        self.mlp_face = nn.Linear(d_model, d_model)
        self.mlp_context = nn.Linear(d_model, d_model)
        self.mlp_text = nn.Linear(d_model, d_model)

        from vie_gameemo.fusion.conv_attention import ConvBranch
        self.conv_branch = ConvBranch(
            in_dim=d_model * n_modalities,
            hidden_dim=d_model,
            n_blocks=n_conv_blocks,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        audio: Tensor,
        face: Tensor,
        context: Tensor,
        text: Tensor,
        has_face: Tensor | None = None,
    ) -> Tensor:
        """Conv-only fusion (no attention branch)."""
        if has_face is not None:
            face = face * has_face.float().view(-1, 1, 1)

        u_a, u_f, u_c, u_t = (self.mlp_audio(audio), self.mlp_face(face),
                               self.mlp_context(context), self.mlp_text(text))
        T = u_a.shape[1]
        for u in (u_f, u_c, u_t):
            if u.shape[1] != T:
                u = F.interpolate(u.transpose(1, 2), T, mode="linear", align_corners=False).transpose(1, 2)
        F_d = torch.cat([u_a, u_f, u_c, u_t], dim=-1)
        return self.conv_branch(F_d)


@register_fusion("attn_only")
class AttnOnly(nn.Module):
    """Ablation: only the attention branch of Conv-Attention 4M."""

    def __init__(self, d_model: int = 768, n_modalities: int = 4, align_to: str = "audio") -> None:
        super().__init__()
        self.align_to = align_to

        self.mlp_audio = nn.Linear(d_model, d_model)
        self.mlp_face = nn.Linear(d_model, d_model)
        self.mlp_context = nn.Linear(d_model, d_model)
        self.mlp_text = nn.Linear(d_model, d_model)

        from vie_gameemo.fusion.conv_attention import AttentionBranch
        self.attn_branch = AttentionBranch(
            in_dim=d_model * n_modalities,
            n_modalities=n_modalities,
        )

    def forward(
        self,
        audio: Tensor,
        face: Tensor,
        context: Tensor,
        text: Tensor,
        has_face: Tensor | None = None,
    ) -> Tensor:
        """Attention-only fusion (no conv branch)."""
        if has_face is not None:
            face = face * has_face.float().view(-1, 1, 1)

        u_a, u_f, u_c, u_t = (self.mlp_audio(audio), self.mlp_face(face),
                               self.mlp_context(context), self.mlp_text(text))
        T = u_a.shape[1]
        u_f = self._align(u_f, T)
        u_c = self._align(u_c, T)
        u_t = self._align(u_t, T)

        F_d = torch.cat([u_a, u_f, u_c, u_t], dim=-1)
        F_s = torch.stack([u_a, u_f, u_c, u_t], dim=-1)
        F_attn, _ = self.attn_branch(F_d, F_s)
        return F_attn

    @staticmethod
    def _align(x: Tensor, T: int) -> Tensor:
        if x.shape[1] == T:
            return x
        if x.shape[1] == 1:
            return x.expand(-1, T, -1)
        return F.interpolate(x.transpose(1, 2), T, mode="linear", align_corners=False).transpose(1, 2)
