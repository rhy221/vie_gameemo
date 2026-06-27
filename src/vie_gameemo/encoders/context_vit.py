"""Webcam context encoder using ViT-ImageNet (Path 2 of dual-path visual).

Encodes the streamer's webcam region (NOT tight face crop) using a generic
ViT pretrained on ImageNet. This captures broader cues than the face encoder:
    - Body language / posture
    - Webcam background / lighting changes
    - Upper-body gestures

Together with Path 1 (face encoder, tight crop), this gives the model both
fine-grained facial expression AND wider streamer context.
"""

import logging

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, ViTModel

logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """Webcam context encoder using generic ViT (Path 2 of dual-path)."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        n_frames: int = 16,
        target_size: tuple[int, int] = (224, 224),
        temporal_pool: str = "mean",
        device: str | torch.device = "cuda",
    ) -> None:
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
    def encode(self, webcam_crops: list[np.ndarray] | None) -> Tensor:
        """Encode webcam region crops (wider than face crops).

        Args:
            webcam_crops: List of BGR ndarrays cropped to the webcam region.
                If None or empty, returns zeros (no-webcam mode).

        Returns:
            Tensor of shape (1, T, 768):
                - T=1 if temporal_pool='mean'
                - T=n_frames if temporal_pool='none'
        """
        if not webcam_crops:
            T = 1 if self.temporal_pool == "mean" else self.n_frames
            return torch.zeros(1, T, 768, device=self.device)

        sampled = _uniform_sample(webcam_crops, self.n_frames)
        cls_tokens = []
        for crop in sampled:
            image = _bgr_to_pil(crop)
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
    def encode_batch(
        self, batch_webcam_crops: list[list[np.ndarray] | None]
    ) -> Tensor:
        """Batch encode multiple clips' webcam regions.

        Args:
            batch_webcam_crops: List of per-clip webcam crop lists (some may be None).

        Returns:
            Tensor of shape (B, T, 768).
        """
        tensors = [self.encode(crops) for crops in batch_webcam_crops]
        return torch.cat(tensors, dim=0)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR ndarray to RGB PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _uniform_sample(items: list, n: int) -> list:
    """Uniformly sample exactly n items, padding with last if needed."""
    if not items:
        return items
    if len(items) >= n:
        indices = np.linspace(0, len(items) - 1, n, dtype=int)
        return [items[i] for i in indices]
    pad_n = n - len(items)
    return list(items) + [items[-1]] * pad_n
