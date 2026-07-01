"""Face encoder using ViT-FER (Path 1 of dual-path visual).

Encodes the streamer's face (cropped from webcam region) using ViT pretrained
on AffectNet for facial expression recognition. Uses a tri-view design:
    - Spatial view: peak frame PATCH tokens (pooled) for micro-expression detail
    - Global view: peak frame CLS token
    - Temporal view: 16 evenly-sampled frames, each CLS token kept

Output layout: [peak_patches | global_CLS | temporal_CLS]

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
    """Tri-view face encoder using ViT-FER (Path 1 of dual-path).

    Output: (B, n_patch_tokens + 1 + n_temporal_frames, 768)
    where n_patch_tokens = spatial_pool[0] * spatial_pool[1].
    """

    def __init__(
        self,
        model_name: str = "trpakov/vit-face-expression",
        n_temporal_frames: int = 16,
        spatial_pool: tuple[int, int] = (4, 4),
        pool_method: str = "mean",
        target_size: tuple[int, int] = (224, 224),
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize face encoder.

        Args:
            model_name: HF model ID (ViT-FER pretrained on AffectNet).
            n_temporal_frames: Frames sampled for temporal view.
            spatial_pool: (H, W) pooling grid on peak frame patches.
            pool_method: How to pool patches: 'mean' | 'max' | 'attention'.
                'attention' weights patches by similarity to the CLS token
                before spatial pooling, highlighting emotion-salient regions.
            target_size: Input frame resize target (W, H).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.n_temporal_frames = n_temporal_frames
        self.spatial_pool = spatial_pool
        self.pool_method = pool_method
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

    @property
    def n_patch_tokens(self) -> int:
        return self.spatial_pool[0] * self.spatial_pool[1]

    @property
    def total_tokens(self) -> int:
        return self.n_patch_tokens + 1 + self.n_temporal_frames

    @torch.no_grad()
    def encode(
        self,
        face_crops: list[np.ndarray] | None,
    ) -> tuple[Tensor, bool]:
        """Encode a sequence of face crops with tri-view design.

        Args:
            face_crops: List of face crop ndarrays (BGR, HxWx3).
                If None or empty, returns zeros (no-facecam mode).

        Returns:
            Tuple of:
                - Tensor (1, T, 768) where T = n_patch + 1 + n_temporal:
                  [peak_patches | global_CLS | temporal_CLS]
                - has_face: True if real face data, False for zeros placeholder
        """
        if not face_crops:
            return torch.zeros(1, self.total_tokens, 768, device=self.device), False

        frames_pil = [_bgr_to_pil(f) for f in face_crops]

        # Peak frame: full hidden states (CLS + patches)
        mid_idx = len(frames_pil) // 2
        peak_cls, peak_patches = self._encode_full_frame(frames_pil[mid_idx])

        # Spatial pool patch tokens — method determined by self.pool_method
        pooled_patches = self._spatial_pool_patches(
            peak_patches, peak_cls, self.spatial_pool[0], self.spatial_pool[1],
        )  # (1, n_patch, 768)

        # Temporal view: n_temporal_frames uniformly sampled
        sampled = _uniform_sample(frames_pil, self.n_temporal_frames)
        temporal_cls = torch.cat(
            [self._encode_single_frame(f) for f in sampled], dim=0
        )  # (n_temporal_frames, 768)
        temporal_cls = temporal_cls.unsqueeze(0)  # (1, n_temporal, 768)

        # [peak_patches | global_CLS | temporal_CLS]
        combined = torch.cat([
            pooled_patches,
            peak_cls.unsqueeze(0),
            temporal_cls,
        ], dim=1)
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
        """Encode one PIL image; return CLS token (1, 768)."""
        inputs = self.processor(images=frame_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token (1, 768)

    def _encode_full_frame(self, frame_pil: Image.Image) -> tuple[Tensor, Tensor]:
        """Encode one PIL image; return CLS token and patch tokens separately.

        Returns:
            Tuple of (CLS (1, 768), patches (1, N_patches, 768)).
        """
        inputs = self.processor(images=frame_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, 1+N_patches, 768)
        return hidden[:, 0, :], hidden[:, 1:, :]  # CLS, patches

    def _spatial_pool_patches(
        self,
        patches: Tensor,
        cls_token: Tensor,
        pool_h: int,
        pool_w: int,
    ) -> Tensor:
        """Spatial pool ViT patch tokens to (B, pool_h*pool_w, D).

        Args:
            patches: Patch tokens (B, N_patches, D) excluding CLS.
            cls_token: CLS token (B, D) — used only for pool_method='attention'.
            pool_h: Target pool grid height.
            pool_w: Target pool grid width.

        Returns:
            Pooled tokens of shape (B, pool_h * pool_w, D).

        pool_method behaviour:
            'mean'      — standard adaptive average pool over the patch grid.
            'max'       — adaptive max pool over the patch grid.
            'attention' — weight each patch by its dot-product similarity with the
                          CLS token before average pooling, surfacing emotion-salient
                          spatial regions without any extra learnable parameters.
        """
        B, N, D = patches.shape
        grid = int(N ** 0.5)

        if self.pool_method == "attention":
            # CLS-guided weighting: attn[b, n] = softmax(CLS_b · patch_bn / sqrt(D))
            scale = D ** 0.5
            q = cls_token.unsqueeze(1)  # (B, 1, D)
            attn = torch.softmax(
                (q @ patches.transpose(1, 2)).squeeze(1) / scale, dim=-1,
            )  # (B, N)
            patches = patches * attn.unsqueeze(-1)  # (B, N, D) weighted
            # Fall through to mean pool on the weighted patches
            pooled = nn.functional.adaptive_avg_pool2d(
                patches.view(B, grid, grid, D).permute(0, 3, 1, 2),
                (pool_h, pool_w),
            )
        elif self.pool_method == "max":
            pooled = nn.functional.adaptive_max_pool2d(
                patches.view(B, grid, grid, D).permute(0, 3, 1, 2),
                (pool_h, pool_w),
            )
        else:  # "mean" (default)
            pooled = nn.functional.adaptive_avg_pool2d(
                patches.view(B, grid, grid, D).permute(0, 3, 1, 2),
                (pool_h, pool_w),
            )

        return pooled.permute(0, 2, 3, 1).reshape(B, pool_h * pool_w, D)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR ndarray to RGB PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _uniform_sample(frames: list, n: int) -> list:
    """Uniformly sample exactly n items from frames list.

    If len(frames) >= n: uniform subsample.
    If len(frames) < n: subsample then pad with last frame to reach exactly n.
    This guarantees the output always has length n, preventing shape mismatches
    between has_face=True and has_face=False paths in the fusion module.
    """
    if not frames:
        return frames
    if len(frames) >= n:
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]
    # Fewer frames than needed — pad by repeating the last frame
    pad_n = n - len(frames)
    return frames + [frames[-1]] * pad_n
