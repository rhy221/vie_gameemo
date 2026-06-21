"""Modal Adapter: projects fusion embeddings into LLM token embedding space.

Implements the modal adapter pattern from Emotion-LLaMAv2 (arXiv:2601.16449),
Section 4.4. Linear projection d_fusion → d_llm enables multimodal features
to be injected as soft tokens directly into the LLM input sequence, eliminating
dependence on annotation text evidence for LLM reasoning.

Trained jointly with LLM LoRA during Stage 2 (cognition). Saved/loaded
alongside cognition checkpoint under key "llm_adapter".
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class ModalAdapter(nn.Module):
    """Linear projection from fusion space to LLM embedding space.

    Projects (B, T, d_fusion) → (B, T, d_llm) so fused multimodal features
    can be injected as soft tokens directly into the LLM's input embedding
    sequence via inputs_embeds, without conversion to text.

    Args:
        d_fusion: Fusion embedding dim (768, output of ConvAttention4M).
        d_llm: LLM hidden dim (4096 for Qwen2.5-7B / LLaMA-2-7B).
    """

    def __init__(self, d_fusion: int = 768, d_llm: int = 4096) -> None:
        super().__init__()
        self.d_fusion = d_fusion
        self.d_llm = d_llm
        self.proj = nn.Linear(d_fusion, d_llm, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Project fusion embedding to LLM token space.

        Args:
            x: (B, T, d_fusion) fused multimodal embedding from ConvAttention4M.

        Returns:
            (B, T, d_llm) soft tokens in LLM embedding space.
        """
        return self.proj(x)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Path,
        d_fusion: int = 768,
        d_llm: int = 4096,
    ) -> "ModalAdapter":
        """Load adapter weights from a cognition checkpoint.

        Args:
            ckpt_path: Path to cognition_best.pt (must contain "llm_adapter" key).
            d_fusion: Source fusion dim (must match checkpoint).
            d_llm: Target LLM dim (must match checkpoint).

        Returns:
            Loaded ModalAdapter instance (on CPU).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        adapter = cls(d_fusion=d_fusion, d_llm=d_llm)
        adapter.load_state_dict(ckpt["llm_adapter"])
        return adapter
