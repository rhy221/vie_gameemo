"""Context encoder using ViT-ImageNet (Path 2 of dual-path visual).

Encodes gameplay context (full frames, NOT cropped to face) using a generic
ViT pretrained on ImageNet. This captures:
    - Game scene state (UI, HUD, kill feed, score)
    - In-game cutscene visuals
    - Overall environmental context

Together with Path 1 (face encoder), this gives the model both streamer's
emotional expression AND the situation triggering it.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, ViTModel

logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """Gameplay context encoder using generic ViT (Path 2 of dual-path)."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        n_frames: int = 16,
        target_size: tuple[int, int] = (224, 224),
        temporal_pool: str = "mean",
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize context encoder.

        Args:
            model_name: HF model ID (ImageNet-pretrained ViT).
            n_frames: Frames sampled per clip for temporal coverage.
            target_size: Frame resize target.
            temporal_pool: 'mean' → single (1,768); 'none' → (n_frames,768).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.n_frames = n_frames
        self.target_size = target_size
        self.temporal_pool = temporal_pool
        self.device = torch.device(device)

        logger.info("Loading context ViT: %s", model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        logger.info("Context encoder loaded and frozen")

    @torch.no_grad()
    def encode(self, frame_paths: list[Path]) -> Tensor:
        """Encode a clip's frames as context.

        Args:
            frame_paths: All extracted frame paths (sorted by time).
                Uniformly subsampled to n_frames.

        Returns:
            Tensor of shape (1, T, 768):
                - T=1 if temporal_pool='mean'
                - T=n_frames if temporal_pool='none'

        Raises:
            ValueError: If frame_paths is empty.
        """
        if not frame_paths:
            T = 1 if self.temporal_pool == "mean" else self.n_frames
            return torch.zeros(1, T, 768, device=self.device)

        sampled = self._sample_frames_uniform(frame_paths, self.n_frames)
        cls_tokens = []
        for fp in sampled:
            image = Image.open(fp).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]  # (1, 768)
            cls_tokens.append(cls)

        stacked = torch.cat(cls_tokens, dim=0)  # (n_frames, 768)

        if self.temporal_pool == "mean":
            return stacked.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, 768)
        else:
            return stacked.unsqueeze(0)  # (1, n_frames, 768)

    @torch.no_grad()
    def encode_batch(self, batch_frame_paths: list[list[Path]]) -> Tensor:
        """Batch encode multiple clips' contexts.

        Args:
            batch_frame_paths: List of per-clip frame path lists.

        Returns:
            Tensor of shape (B, T, 768).
        """
        tensors = [self.encode(fps) for fps in batch_frame_paths]
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _sample_frames_uniform(frame_paths: list[Path], n: int) -> list[Path]:
        """Uniformly subsample exactly n frames from a list, preserving order.

        Args:
            frame_paths: Sorted list of frame paths.
            n: Target count.

        Returns:
            List of exactly n paths. If len < n, last frame is repeated.
        """
        if not frame_paths:
            return frame_paths
        if len(frame_paths) >= n:
            indices = np.linspace(0, len(frame_paths) - 1, n, dtype=int)
            return [frame_paths[i] for i in indices]
        pad_n = n - len(frame_paths)
        return list(frame_paths) + [frame_paths[-1]] * pad_n
