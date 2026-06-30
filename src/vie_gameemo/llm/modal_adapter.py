"""Modal Adapter: projects fusion + raw modality embeddings into LLM space.

Follows the multi-stream projection pattern from Emotion-LLaMAv2
(arXiv:2601.16449): each raw modality gets its own Linear projection into
LLM token space, and the outputs are concatenated alongside the fusion
projection to form the full soft-token sequence for the LLM.

This gives the LLM direct access to both the fused representation AND
per-modality raw features — matching v2's 448-token approach instead of
collapsing everything into a single mean-pooled token.

Trained jointly with LLM LoRA during Stage 2 (cognition). Saved/loaded
alongside cognition checkpoint under key "llm_adapter".
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class ModalAdapter(nn.Module):
    """Multi-stream projection from fusion + raw modalities to LLM space.

    Projects fusion output AND each raw modality separately into LLM
    embedding space, then concatenates along the sequence dimension.
    Generates attention masks to zero out missing modalities (e.g. face).

    Output token layout (along seq dim):
        [fusion_tokens | audio_tokens | face_tokens | context_tokens | text_tokens]

    Args:
        d_fusion: Fusion embedding dim (768, output of ConvAttention4M).
        d_modality: Per-modality encoder output dim (768 by default).
        d_llm: LLM hidden dim (4096 for Qwen2.5-7B / LLaMA-2-7B).
    """

    def __init__(
        self,
        d_fusion: int = 768,
        d_modality: int = 768,
        d_llm: int = 4096,
        d_penult: int = 256,
    ) -> None:
        super().__init__()
        self.d_fusion = d_fusion
        self.d_modality = d_modality
        self.d_llm = d_llm
        self.d_penult = d_penult

        self.proj_penult = nn.Linear(d_penult, d_llm)
        self.proj_fusion = nn.Linear(d_fusion, d_llm)
        self.proj_audio = nn.Linear(d_modality, d_llm)
        self.proj_face = nn.Linear(d_modality, d_llm)
        self.proj_context = nn.Linear(d_modality, d_llm)
        self.proj_text = nn.Linear(d_modality, d_llm)

    def forward(
        self,
        fusion_emb: Tensor,
        penult: Tensor | None = None,
        audio: Tensor | None = None,
        face: Tensor | None = None,
        context: Tensor | None = None,
        text: Tensor | None = None,
        has_face: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Project fusion + raw modalities to LLM token space.

        When raw modality tensors are not provided, falls back to
        fusion-only projection (backward compatible with old callers).

        Args:
            fusion_emb: (B, T_f, d_fusion) fused output from ConvAttention4M.
            penult: (B, 256) MLP penultimate vector. If provided, projected
                and prepended as first soft token (faithfulness anchor).
            audio: (B, T_a, d_modality) raw audio encoder output.
            face: (B, T_face, d_modality) raw face encoder output.
            context: (B, T_c, d_modality) raw context encoder output.
            text: (B, T_t, d_modality) raw text encoder output.
            has_face: (B,) bool tensor. False → face tokens are masked out.

        Returns:
            Tuple of:
                - soft_tokens: (B, T_total, d_llm) concatenated projections.
                - attn_mask: (B, T_total) attention mask (1=attend, 0=ignore).
        """
        B = fusion_emb.shape[0]
        device = fusion_emb.device

        # Penult token (faithfulness anchor from MLP classifier)
        parts: list[Tensor] = []
        mask_parts: list[Tensor] = []

        if penult is not None:
            if penult.dim() == 2:
                penult = penult.unsqueeze(1)  # (B, 256) → (B, 1, 256)
            penult_tok = self.proj_penult(penult)  # (B, 1, d_llm)
            parts.append(penult_tok)
            mask_parts.append(torch.ones(B, penult_tok.shape[1], dtype=torch.long, device=device))

        fusion_tok = self.proj_fusion(fusion_emb)

        raw_modalities = [audio, face, context, text]
        if all(m is None for m in raw_modalities) and penult is None:
            mask = torch.ones(B, fusion_tok.shape[1], dtype=torch.long, device=device)
            return fusion_tok, mask

        parts.append(fusion_tok)
        mask_parts.append(
            torch.ones(B, fusion_tok.shape[1], dtype=torch.long, device=device),
        )

        proj_map = [
            (audio, self.proj_audio, None),
            (face, self.proj_face, has_face),
            (context, self.proj_context, None),
            (text, self.proj_text, None),
        ]

        for feat, proj, modality_mask in proj_map:
            if feat is None:
                continue
            projected = proj(feat)
            parts.append(projected)

            # Auto-detect all-zeros modalities (like v2) + explicit mask (face)
            all_zero = feat.abs().sum(dim=list(range(1, feat.dim()))) == 0  # (B,)
            m = torch.ones(B, projected.shape[1], dtype=torch.long, device=device)
            m[all_zero] = 0
            if modality_mask is not None:
                m = m * modality_mask.long().unsqueeze(-1)
            mask_parts.append(m)

        soft_tokens = torch.cat(parts, dim=1)
        attn_mask = torch.cat(mask_parts, dim=1)

        return soft_tokens, attn_mask

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Path,
        d_fusion: int = 768,
        d_modality: int = 768,
        d_llm: int = 4096,
        d_penult: int = 256,
    ) -> "ModalAdapter":
        """Load adapter weights from a cognition checkpoint.

        Args:
            ckpt_path: Path to checkpoint (must contain "llm_adapter" key).
            d_fusion: Source fusion dim (must match checkpoint).
            d_modality: Per-modality dim (must match checkpoint).
            d_llm: Target LLM dim (must match checkpoint).
            d_penult: MLP penultimate dim (must match checkpoint).

        Returns:
            Loaded ModalAdapter instance (on CPU).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        adapter = cls(d_fusion=d_fusion, d_modality=d_modality, d_llm=d_llm, d_penult=d_penult)
        adapter.load_state_dict(ckpt["llm_adapter"], strict=False)
        return adapter
