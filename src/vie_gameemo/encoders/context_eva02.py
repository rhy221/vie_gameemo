"""Context encoder using EVA-02-B (Path 2, EVA-02 branch).

EVA-02-B: ViT-B with patch_size=14, RoPE + SwiGLU, pretrained on ImageNet-22k
via Masked Image Modeling. At 224×224 it produces 256 patch tokens (16×16 grid)
vs ViT-B/16's 196 (14×14), giving finer-grained spatial features.
Output D=768 matches the project-wide d_model convention.

Same tri-view design as ContextEncoder (context_vit.py):
  [peak_patches | global_CLS | temporal_patches_flat]
  (B, n_patch_tokens + 1 + n_frames * temporal_n_patch_tokens, 768)

Requires: timm >= 0.9  (pip install timm)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn

# Re-use utility functions from context_vit to avoid duplication.
from vie_gameemo.encoders.context_vit import _bgr_to_pil, _uniform_sample

logger = logging.getLogger(__name__)

# Standard ImageNet normalization (matches mim_in22k checkpoint).
_EVA02_MEAN = (0.485, 0.456, 0.406)
_EVA02_STD  = (0.229, 0.224, 0.225)


def _build_transform(target_size: tuple[int, int]) -> T.Compose:
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=_EVA02_MEAN, std=_EVA02_STD),
    ])


class Eva02ContextEncoder(nn.Module):
    """Tri-view context encoder using EVA-02-B (timm backbone).

    Drop-in replacement for ContextEncoder: identical tri-view design,
    identical output shape (B, n_patch + 1 + n_frames*t_n_patch, 768).
    Differs only in backbone — EVA-02-B gives 256 patches/frame vs 196
    for ViT-B/16, so spatial pooling has a finer input grid (16×16 vs 14×14).

    Output: (B, n_patch_tokens + 1 + n_frames * temporal_n_patch_tokens, 768)
    where n_patch_tokens          = spatial_pool[0] * spatial_pool[1]
    and   temporal_n_patch_tokens = temporal_spatial_pool[0] * temporal_spatial_pool[1].
    """

    def __init__(
        self,
        model_name: str = "eva02_base_patch14_224.mim_in22k",
        n_frames: int = 16,
        spatial_pool: tuple[int, int] = (2, 2),
        temporal_spatial_pool: tuple[int, int] = (2, 2),
        pool_method: str = "mean",
        target_size: tuple[int, int] = (224, 224),
        use_temporal_3d_pool: bool = False,
        temporal_3d_pool: tuple[int, int, int] = (4, 4, 4),
        device: str | torch.device = "cuda",
    ) -> None:
        """
        Args:
            model_name: timm model string (see timm.list_models('eva02*')).
            n_frames: Frames sampled for temporal view.
            spatial_pool: (H, W) pooling grid for the peak frame global view.
            temporal_spatial_pool: (H, W) pooling grid applied per temporal frame.
                (2, 2) → 4 tokens/frame × n_frames = 64 tokens (paper default).
            pool_method: 'mean' | 'max' | 'attention'.
            target_size: Resize target (H, W) before patch embedding.
            use_temporal_3d_pool: If True, pool all temporal frames jointly in 3D
                (T, H, W) instead of per-frame 2D.
            temporal_3d_pool: (T', H', W') output grid when use_temporal_3d_pool=True.
                Default (4,4,4) → 64 tokens.
            device: Torch device string or object.
        """
        super().__init__()
        try:
            import timm  # noqa: F401 — checked here so the error is early and clear
        except ImportError as exc:
            raise ImportError(
                "Eva02ContextEncoder requires timm. "
                "Install with: pip install timm"
            ) from exc

        import timm as _timm

        self.model_name = model_name
        self.n_frames = n_frames
        self.spatial_pool = spatial_pool
        self.temporal_spatial_pool = temporal_spatial_pool
        self.pool_method = pool_method
        self.target_size = target_size
        self.use_temporal_3d_pool = use_temporal_3d_pool
        self.temporal_3d_pool = temporal_3d_pool
        self.device = torch.device(device)

        logger.info("Loading EVA-02-B via timm: %s", model_name)
        self.model = _timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        logger.info("EVA-02-B loaded and frozen")

        self.transform = _build_transform(target_size)

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def n_patch_tokens(self) -> int:
        return self.spatial_pool[0] * self.spatial_pool[1]

    @property
    def temporal_n_patch_tokens(self) -> int:
        return self.temporal_spatial_pool[0] * self.temporal_spatial_pool[1]

    @property
    def total_tokens(self) -> int:
        if self.use_temporal_3d_pool:
            t_tokens = self.temporal_3d_pool[0] * self.temporal_3d_pool[1] * self.temporal_3d_pool[2]
        else:
            t_tokens = self.n_frames * self.temporal_n_patch_tokens
        return self.n_patch_tokens + 1 + t_tokens

    # ------------------------------------------------------------------
    # Public encode API (mirrors ContextEncoder)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, webcam_crops: list[np.ndarray] | None) -> Tensor:
        """Encode webcam region crops with tri-view design.

        Args:
            webcam_crops: List of BGR ndarrays. None/empty → zero tensor.

        Returns:
            (1, n_patch + 1 + n_frames*t_n_patch, 768)
        """
        if not webcam_crops:
            return torch.zeros(1, self.total_tokens, 768, device=self.device)

        frames_pil = [_bgr_to_pil(f) for f in webcam_crops]

        mid_idx = len(frames_pil) // 2
        global_cls, peak_patches = self._encode_full_frame(frames_pil[mid_idx])

        pooled_patches = self._spatial_pool_patches(
            peak_patches, global_cls, self.spatial_pool[0], self.spatial_pool[1],
        )

        sampled = _uniform_sample(frames_pil, self.n_frames)
        if self.use_temporal_3d_pool:
            temporal = self._temporal_3d_pool(sampled)
        else:
            temporal_parts = []
            for f in sampled:
                frame_cls, frame_patches = self._encode_full_frame(f)
                pooled = self._spatial_pool_patches(
                    frame_patches, frame_cls,
                    self.temporal_spatial_pool[0], self.temporal_spatial_pool[1],
                )
                temporal_parts.append(pooled)
            temporal = torch.cat(temporal_parts, dim=1)

        return torch.cat([pooled_patches, global_cls.unsqueeze(0), temporal], dim=1)

    @torch.no_grad()
    def encode_from_paths(self, frame_paths: list[Path]) -> Tensor:
        """Encode full frames from file paths (fallback when no webcam detected).

        Args:
            frame_paths: Paths to extracted frame images.

        Returns:
            (1, n_patch + 1 + n_frames*t_n_patch, 768)
        """
        if not frame_paths:
            return torch.zeros(1, self.total_tokens, 768, device=self.device)

        paths_list = list(frame_paths)
        sampled_paths = _uniform_sample(paths_list, self.n_frames)
        frames_pil = [Image.open(fp).convert("RGB") for fp in sampled_paths]

        mid_idx = len(frames_pil) // 2
        global_cls, peak_patches = self._encode_full_frame(frames_pil[mid_idx])

        pooled_patches = self._spatial_pool_patches(
            peak_patches, global_cls, self.spatial_pool[0], self.spatial_pool[1],
        )

        if self.use_temporal_3d_pool:
            temporal = self._temporal_3d_pool(frames_pil)
        else:
            temporal_parts = []
            for f in frames_pil:
                frame_cls, frame_patches = self._encode_full_frame(f)
                pooled = self._spatial_pool_patches(
                    frame_patches, frame_cls,
                    self.temporal_spatial_pool[0], self.temporal_spatial_pool[1],
                )
                temporal_parts.append(pooled)
            temporal = torch.cat(temporal_parts, dim=1)

        return torch.cat([pooled_patches, global_cls.unsqueeze(0), temporal], dim=1)

    @torch.no_grad()
    def encode_batch(
        self, batch_webcam_crops: list[list[np.ndarray] | None]
    ) -> Tensor:
        """Batch encode multiple clips. Returns (B, total_tokens, 768)."""
        return torch.cat([self.encode(crops) for crops in batch_webcam_crops], dim=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _temporal_3d_pool(self, frames_pil: list) -> Tensor:
        """Pool all temporal frames jointly in 3D (T, H, W) → (T'*H'*W', D)."""
        all_patches = []
        for f in frames_pil:
            _, frame_patches = self._encode_full_frame(f)  # (1, N, D)
            all_patches.append(frame_patches)

        T = len(all_patches)
        patches_stack = torch.cat(all_patches, dim=0)  # (T, N, D)
        _, N, D = patches_stack.shape
        grid = int(N ** 0.5)

        vol = patches_stack.view(T, grid, grid, D)            # (T, H, W, D)
        vol = vol.permute(3, 0, 1, 2).unsqueeze(0)           # (1, D, T, H, W)
        td, th, tw = self.temporal_3d_pool
        pooled = F.adaptive_avg_pool3d(vol, (td, th, tw))    # (1, D, td, th, tw)
        return pooled.permute(0, 2, 3, 4, 1).reshape(1, td * th * tw, D)

    def _encode_full_frame(self, frame_pil: Image.Image) -> tuple[Tensor, Tensor]:
        """Run EVA-02-B on one PIL image. Returns (CLS (1,768), patches (1,N,768)).

        EVA-02-B patch_size=14 → 256 patches at 224×224 (16×16 grid).
        timm forward_features returns (B, 1+N, D); [0] is CLS, [1:] are patches.
        """
        x = self.transform(frame_pil).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        hidden = self.model.forward_features(x)                      # (1, 1+N, 768)
        if hidden.ndim != 3:
            raise RuntimeError(
                f"EVA-02 forward_features returned shape {hidden.shape}; "
                "expected (B, 1+N_patches, D). Upgrade timm: pip install -U timm"
            )
        return hidden[:, 0, :], hidden[:, 1:, :]  # CLS, patches

    def _spatial_pool_patches(
        self,
        patches: Tensor,
        cls_token: Tensor,
        pool_h: int,
        pool_w: int,
    ) -> Tensor:
        """Spatial pool patch tokens to (B, pool_h*pool_w, D).

        Works for any square patch grid — 14×14 (ViT-B/16) or 16×16 (EVA-02-B/14)
        — because grid size is inferred from N via int(N**0.5).

        pool_method:
            'mean'      — adaptive average pool.
            'max'       — adaptive max pool.
            'attention' — CLS-guided weighted average pool (no extra params).
        """
        B, N, D = patches.shape
        grid = int(N ** 0.5)

        if self.pool_method == "attention":
            scale = D ** 0.5
            q = cls_token.unsqueeze(1)  # (B, 1, D)
            attn = torch.softmax(
                (q @ patches.transpose(1, 2)).squeeze(1) / scale, dim=-1,
            )  # (B, N)
            patches = patches * attn.unsqueeze(-1)
            pooled = nn.functional.adaptive_avg_pool2d(
                patches.view(B, grid, grid, D).permute(0, 3, 1, 2),
                (pool_h, pool_w),
            )
        elif self.pool_method == "max":
            pooled = nn.functional.adaptive_max_pool2d(
                patches.view(B, grid, grid, D).permute(0, 3, 1, 2),
                (pool_h, pool_w),
            )
        else:  # "mean"
            pooled = nn.functional.adaptive_avg_pool2d(
                patches.view(B, grid, grid, D).permute(0, 3, 1, 2),
                (pool_h, pool_w),
            )

        return pooled.permute(0, 2, 3, 1).reshape(B, pool_h * pool_w, D)
