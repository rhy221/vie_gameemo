"""Context encoder using ViT-ImageNet (Path 2 of dual-path visual).

Tri-view design (mirrors FaceEncoder) to ensure both global features and
video temporal features are captured:
    - Spatial view: peak (middle) frame PATCH tokens, spatially pooled
    - Global view: peak frame CLS token
    - Temporal view: n_frames evenly-sampled frames, each CLS token kept

Output layout: [peak_patches | global_CLS | temporal_CLS]
Output shape: (B, n_patch_tokens + 1 + n_temporal_frames, 768)

Prefers webcam region crops (body language, posture, background) when a
webcam is detected. Falls back to full frames when no webcam is found,
so the context modality is never all-zeros.
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


class ContextEncoder(nn.Module):
    """Tri-view webcam context encoder using generic ViT (Path 2 of dual-path).

    Mirrors FaceEncoder's tri-view design:
        - Spatial patches from peak frame capture local gameplay detail
          (UI elements, health bars, kill feed regions).
        - Global CLS from peak frame captures overall scene semantics
          (combat vs menu vs cutscene).
        - Temporal CLS sequence captures scene changes over the 5-second clip
          (calm → sudden action → reaction).

    Output: (B, n_patch_tokens + 1 + n_temporal_frames, 768)
    where n_patch_tokens = spatial_pool[0] * spatial_pool[1].
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        n_frames: int = 16,
        spatial_pool: tuple[int, int] = (2, 2),
        pool_method: str = "mean",
        target_size: tuple[int, int] = (224, 224),
        temporal_pool: str | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize context encoder.

        Args:
            model_name: HF model ID (ViT pretrained on ImageNet-21k).
            n_frames: Frames sampled for temporal view.
            spatial_pool: (H, W) pooling grid on peak frame patches.
            pool_method: How to pool patches: 'mean' | 'max' | 'attention'.
                'attention' weights patches by similarity to the peak CLS token
                before spatial pooling (no learnable parameters required).
            target_size: Input frame resize target (W, H).
            temporal_pool: Deprecated parameter kept for backward compatibility;
                ignored in tri-view design (temporal CLS tokens are always kept).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.n_frames = n_frames
        self.spatial_pool = spatial_pool
        self.pool_method = pool_method
        self.target_size = target_size
        self.device = torch.device(device)

        if temporal_pool is not None:
            logger.warning(
                "ContextEncoder: 'temporal_pool' is deprecated and ignored "
                "in tri-view design. Temporal CLS tokens are always preserved."
            )

        logger.info("Loading context ViT: %s", model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        logger.info("Context encoder loaded and frozen")

    @property
    def n_patch_tokens(self) -> int:
        return self.spatial_pool[0] * self.spatial_pool[1]

    @property
    def total_tokens(self) -> int:
        return self.n_patch_tokens + 1 + self.n_frames

    @torch.no_grad()
    def encode(self, webcam_crops: list[np.ndarray] | None) -> Tensor:
        """Encode webcam region crops with tri-view design.

        Args:
            webcam_crops: List of BGR ndarrays cropped to the webcam region.
                If None or empty, returns zeros (no-webcam mode).

        Returns:
            Tensor of shape (1, n_patch_tokens + 1 + n_temporal_frames, 768):
                [peak_patches | global_CLS | temporal_CLS]
        """
        if not webcam_crops:
            return torch.zeros(1, self.total_tokens, 768, device=self.device)

        frames_pil = [_bgr_to_pil(f) for f in webcam_crops]

        # Peak frame: full hidden states (CLS + patches)
        mid_idx = len(frames_pil) // 2
        global_cls, peak_patches = self._encode_full_frame(frames_pil[mid_idx])

        # Spatial pool patch tokens for local detail
        pooled_patches = self._spatial_pool_patches(
            peak_patches, global_cls, self.spatial_pool[0], self.spatial_pool[1],
        )  # (1, n_patch, 768)

        # Temporal view: n_frames uniformly sampled, each CLS kept separately
        sampled = _uniform_sample(frames_pil, self.n_frames)
        temporal_cls = torch.cat(
            [self._encode_single_frame(f) for f in sampled], dim=0
        )  # (n_frames, 768)
        temporal_cls = temporal_cls.unsqueeze(0)  # (1, n_frames, 768)

        # [peak_patches | global_CLS | temporal_CLS]
        combined = torch.cat([
            pooled_patches,
            global_cls.unsqueeze(0),
            temporal_cls,
        ], dim=1)
        return combined  # (1, total_tokens, 768)

    @torch.no_grad()
    def encode_from_paths(self, frame_paths: list[Path]) -> Tensor:
        """Encode full frames from file paths (fallback when no webcam detected).

        Args:
            frame_paths: Paths to extracted frame images.

        Returns:
            Tensor of shape (1, n_patch_tokens + 1 + n_temporal_frames, 768).
        """
        if not frame_paths:
            return torch.zeros(1, self.total_tokens, 768, device=self.device)

        paths_list = list(frame_paths)
        sampled_paths = _uniform_sample(paths_list, self.n_frames)
        frames_pil = [Image.open(fp).convert("RGB") for fp in sampled_paths]

        # Peak frame from the sampled set
        mid_idx = len(frames_pil) // 2
        global_cls, peak_patches = self._encode_full_frame(frames_pil[mid_idx])

        pooled_patches = self._spatial_pool_patches(
            peak_patches, global_cls, self.spatial_pool[0], self.spatial_pool[1],
        )

        temporal_cls = torch.cat(
            [self._encode_single_frame(f) for f in frames_pil], dim=0
        ).unsqueeze(0)  # (1, n_frames, 768)

        return torch.cat([pooled_patches, global_cls.unsqueeze(0), temporal_cls], dim=1)

    @torch.no_grad()
    def encode_batch(
        self, batch_webcam_crops: list[list[np.ndarray] | None]
    ) -> Tensor:
        """Batch encode multiple clips' webcam regions.

        Args:
            batch_webcam_crops: List of per-clip webcam crop lists (some may be None).

        Returns:
            Tensor of shape (B, total_tokens, 768).
        """
        tensors = [self.encode(crops) for crops in batch_webcam_crops]
        return torch.cat(tensors, dim=0)

    def _encode_single_frame(self, frame_pil: Image.Image) -> Tensor:
        """Encode one PIL image; return CLS token (1, 768)."""
        inputs = self.processor(images=frame_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token (1, 768)

    def _encode_full_frame(self, frame_pil: Image.Image) -> tuple[Tensor, Tensor]:
        """Encode one PIL image; return CLS and patch tokens separately.

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
                          CLS token before average pooling, surfacing salient
                          spatial regions without any extra learnable parameters.
        """
        B, N, D = patches.shape
        grid = int(N ** 0.5)

        if self.pool_method == "attention":
            scale = D ** 0.5
            q = cls_token.unsqueeze(1)  # (B, 1, D)
            attn = torch.softmax(
                (q @ patches.transpose(1, 2)).squeeze(1) / scale, dim=-1,
            )  # (B, N)
            patches = patches * attn.unsqueeze(-1)  # (B, N, D) weighted
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


def _uniform_sample(items: list, n: int) -> list:
    """Uniformly sample exactly n items, padding with last if needed."""
    if not items:
        return items
    if len(items) >= n:
        indices = np.linspace(0, len(items) - 1, n, dtype=int)
        return [items[i] for i in indices]
    pad_n = n - len(items)
    return list(items) + [items[-1]] * pad_n
