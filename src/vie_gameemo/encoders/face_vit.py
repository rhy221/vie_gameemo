"""Face encoder using ViT-FER (Path 1 of dual-path visual).

Encodes the streamer's face (cropped from webcam region) using ViT pretrained
on AffectNet for facial expression recognition. Uses a dual-view design:
    - Global view: middle frame CLS token
    - Temporal view: 16 evenly-sampled frames, each CLS token kept

IMPORTANT: This encoder receives FACE CROPS from face_crop.py, not full
frames. Do not feed full frames into ViT-FER (distribution mismatch with
AffectNet pretrain).
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, ViTModel

logger = logging.getLogger(__name__)


class FaceEncoder(nn.Module):
    """Dual-view face encoder using ViT-FER (Path 1 of dual-path)."""

    def __init__(
        self,
        model_name: str = "trpakov/vit-face-expression",
        n_temporal_frames: int = 16,
        spatial_pool: tuple[int, int] = (2, 2),
        target_size: tuple[int, int] = (224, 224),
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize face encoder.

        Args:
            model_name: HF model ID (ViT-FER pretrained on AffectNet).
            n_temporal_frames: Frames sampled for temporal view.
            spatial_pool: (H, W) pooling on patch grid.
            target_size: Input frame resize target (W, H).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.n_temporal_frames = n_temporal_frames
        self.spatial_pool = spatial_pool
        self.target_size = target_size
        self.device = torch.device(device)

        logger.info("Loading ViT-FER: %s", model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        logger.info("Face encoder loaded and frozen")

    @torch.no_grad()
    def encode(
        self,
        face_crops: list[np.ndarray] | None,
    ) -> tuple[Tensor, bool]:
        """Encode a sequence of face crops with dual-view design.

        Args:
            face_crops: List of face crop ndarrays (BGR, HxWx3).
                If None or empty, returns zeros (no-facecam mode).

        Returns:
            Tuple of:
                - Tensor (1, T, 768): global CLS token followed by temporal CLS tokens
                - has_face: True if real face data, False for zeros placeholder
        """
        if not face_crops:
            T = 1 + self.n_temporal_frames
            return torch.zeros(1, T, 768, device=self.device), False

        frames_pil = [_bgr_to_pil(f) for f in face_crops]

        # Global view: middle frame
        mid_idx = len(frames_pil) // 2
        global_cls = self._encode_single_frame(frames_pil[mid_idx])  # (1, 768)

        # Temporal view: n_temporal_frames uniformly sampled
        sampled = _uniform_sample(frames_pil, self.n_temporal_frames)
        temporal_cls = torch.cat(
            [self._encode_single_frame(f) for f in sampled], dim=0
        )  # (n_temporal_frames, 768)
        temporal_cls = temporal_cls.unsqueeze(0)  # (1, n_temporal, 768)

        combined = torch.cat([global_cls.unsqueeze(0), temporal_cls], dim=1)  # (1, 1+n_t, 768)
        return combined, True

    @torch.no_grad()
    def encode_batch(
        self,
        batch_face_crops: list[list[np.ndarray] | None],
    ) -> tuple[Tensor, list[bool]]:
        """Batch encode multiple clips.

        Args:
            batch_face_crops: List of per-clip face crop lists (some may be None).

        Returns:
            Tuple of (stacked tensor (B, T, 768), list of has_face flags).
        """
        tensors = []
        flags = []
        for crops in batch_face_crops:
            t, flag = self.encode(crops)
            tensors.append(t)
            flags.append(flag)
        return torch.cat(tensors, dim=0), flags

    def _encode_single_frame(self, frame_pil: Image.Image) -> Tensor:
        """Encode one PIL image; return CLS token (1, 768).

        Args:
            frame_pil: RGB PIL image.

        Returns:
            CLS token tensor of shape (1, 768).
        """
        inputs = self.processor(images=frame_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token (1, 768)

    @staticmethod
    def _spatial_pool_patches(
        patches: Tensor,
        pool_h: int,
        pool_w: int,
    ) -> Tensor:
        """Spatial average pool over ViT patch tokens.

        Args:
            patches: Patch tokens (B, N_patches, D) excluding CLS.
            pool_h: Target pool grid height.
            pool_w: Target pool grid width.

        Returns:
            Pooled tokens of shape (B, pool_h * pool_w, D).
        """
        B, N, D = patches.shape
        grid = int(N ** 0.5)
        patches_grid = patches.view(B, grid, grid, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        pooled = nn.functional.adaptive_avg_pool2d(patches_grid, (pool_h, pool_w))
        return pooled.permute(0, 2, 3, 1).reshape(B, pool_h * pool_w, D)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR ndarray to RGB PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _uniform_sample(frames: list, n: int) -> list:
    """Uniformly sample n items from frames list."""
    if len(frames) <= n:
        return frames
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]
