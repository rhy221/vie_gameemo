"""Context encoder using ViT-ImageNet or EVA ViT-B (Path 2 of dual-path visual).

Prefers webcam region crops (body language, posture, background) when a
webcam is detected. Falls back to full frames when no webcam is found,
so the context modality is never all-zeros.

Two backbones are supported via ``backend``:
    - "vit"        (default): plain HF ViTModel, e.g. google/vit-base-patch16-224.
      Feature = CLS token of last_hidden_state.
    - "eva_vit_b": EVA ViT-B, loaded through transformers' timm-wrapper
      (requires the `timm` package). Feature = pooler_output if the wrapper
      exposes one (global pool over patch tokens), else mean over
      last_hidden_state.
Both backbones are loaded via AutoModel/AutoImageProcessor so either can be
swapped in purely via config — no code branching is needed downstream
(fusion consumes whatever `d_out` the encoder reports).

``temporal_pool`` and ``use_patch_tokens`` control how much of the 16 sampled
frames survives into the fused representation (see Emotion-LLaMA-v2's
spatiotemporal-downsampled "video" stream for the design this mirrors):
    - temporal_pool="mean" (previous default): collapse across frames.
    - temporal_pool="none": keep all n_frames steps (no time collapse).
    - use_patch_tokens=False (previous default, "vit" backend only otherwise
      ignored): 1 CLS token/frame.
    - use_patch_tokens=True ("vit" backend only): spatial_pool[0]*spatial_pool[1]
      pooled patch tokens/frame instead of just CLS, preserving a coarse
      spatial layout (posture/position), not just "eva_vit_b" appearance.
Both are independent, off-by-default toggles — set both back to
("mean", False) to fall back to the original single-mean-token behavior if
the richer representation doesn't help.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """Webcam context encoder using a generic ViT-family backbone (Path 2 of dual-path)."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        backend: str = "vit",
        n_frames: int = 16,
        target_size: tuple[int, int] = (224, 224),
        temporal_pool: str = "mean",
        use_patch_tokens: bool = False,
        spatial_pool: tuple[int, int] = (2, 2),
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backend = backend
        self.n_frames = n_frames
        self.target_size = target_size
        self.temporal_pool = temporal_pool
        self.spatial_pool = spatial_pool
        self.use_patch_tokens = use_patch_tokens
        if use_patch_tokens and backend != "vit":
            logger.warning(
                "use_patch_tokens=True only supported for backend='vit' (needs "
                "a known CLS+patch last_hidden_state layout); falling back to "
                "CLS-only for backend=%r",
                backend,
            )
            self.use_patch_tokens = False
        self.device = torch.device(device)

        logger.info("Loading context encoder (backend=%s): %s", backend, model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)

        self.d_out = self._probe_output_dim()
        logger.info(
            "Context encoder loaded and frozen (d_out=%d, temporal_pool=%s, "
            "use_patch_tokens=%s, tokens/clip=%d)",
            self.d_out, self.temporal_pool, self.use_patch_tokens, self._output_seq_len(),
        )

    @torch.no_grad()
    def _probe_output_dim(self) -> int:
        """Run one dummy forward pass to determine the embedding width.

        Backends expose their output dim under different config attributes
        (hidden_size for HF ViT, num_features for a timm-wrapped model), so
        probing empirically is more robust than reading a specific attribute
        name — it also validates the extraction path (CLS vs pooler) works
        end-to-end at load time instead of failing later mid-training.
        """
        dummy = Image.new("RGB", self.target_size)
        inputs = self.processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return int(self._extract_embedding(outputs).shape[-1])

    def _extract_embedding(self, outputs) -> Tensor:
        """Extract a single (1, D) embedding per frame from a model forward pass."""
        if self.backend == "eva_vit_b":
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state.mean(dim=1)
        return outputs.last_hidden_state[:, 0, :]

    def _tokens_per_frame(self) -> int:
        return self.spatial_pool[0] * self.spatial_pool[1] if self.use_patch_tokens else 1

    def _output_seq_len(self) -> int:
        per_frame = self._tokens_per_frame()
        return per_frame if self.temporal_pool == "mean" else self.n_frames * per_frame

    def _encode_frame(self, image: Image.Image) -> Tensor:
        """Encode one frame → (tokens_per_frame, d_out).

        CLS-only mode (default) returns (1, d_out). Patch-token mode returns
        (spatial_pool[0]*spatial_pool[1], d_out) — pooled patch grid instead
        of just the CLS summary, preserving a coarse spatial layout.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        if self.use_patch_tokens:
            patches = outputs.last_hidden_state[:, 1:, :]  # (1, N, D), drop CLS
            return self._spatial_pool_patches(patches, *self.spatial_pool)[0]  # (n_patch, D)
        return self._extract_embedding(outputs)  # (1, D) — batch dim doubles as "1 token"

    def _encode_sequence(self, frames: list) -> Tensor:
        """Encode a list of already-sampled frames (PIL Images) into (1, T, d_out).

        T depends on temporal_pool/use_patch_tokens (see _output_seq_len).
        """
        # Each _encode_frame call returns (tokens_per_frame, D) — stacking gives
        # (n_frames, tokens_per_frame, D) uniformly, whether tokens_per_frame
        # is 1 (CLS-only) or spatial_pool[0]*spatial_pool[1] (patch tokens).
        per_frame = torch.stack([self._encode_frame(f) for f in frames], dim=0)

        if self.temporal_pool == "mean":
            pooled = per_frame.mean(dim=0)  # (tokens_per_frame, D) — collapse time, keep space
        else:
            pooled = per_frame.reshape(-1, per_frame.shape[-1])  # (n_frames*tokens_per_frame, D)
        return pooled.unsqueeze(0)  # (1, T, D)

    @staticmethod
    def _spatial_pool_patches(patches: Tensor, pool_h: int, pool_w: int) -> Tensor:
        """Spatial average pool over ViT patch tokens (excludes CLS).

        Args:
            patches: (B, N_patches, D), N_patches assumed to form a square grid.
            pool_h, pool_w: Target pool grid size.

        Returns:
            (B, pool_h*pool_w, D) pooled tokens.
        """
        B, N, D = patches.shape
        grid = int(N ** 0.5)
        patches_grid = patches.view(B, grid, grid, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        pooled = nn.functional.adaptive_avg_pool2d(patches_grid, (pool_h, pool_w))
        return pooled.permute(0, 2, 3, 1).reshape(B, pool_h * pool_w, D)

    @torch.no_grad()
    def encode(self, webcam_crops: list[np.ndarray] | None) -> Tensor:
        """Encode webcam region crops (wider than face crops).

        Args:
            webcam_crops: List of BGR ndarrays cropped to the webcam region.
                If None or empty, returns zeros (no-webcam mode).

        Returns:
            Tensor of shape (1, T, d_out); T = `_output_seq_len()`.
        """
        if not webcam_crops:
            return torch.zeros(1, self._output_seq_len(), self.d_out, device=self.device)

        sampled = _uniform_sample(webcam_crops, self.n_frames)
        frames = [_bgr_to_pil(crop) for crop in sampled]
        return self._encode_sequence(frames)

    @torch.no_grad()
    def encode_from_paths(self, frame_paths: list[Path]) -> Tensor:
        """Encode full frames from file paths (fallback when no webcam detected).

        Args:
            frame_paths: Paths to extracted frame images.

        Returns:
            Tensor of shape (1, T, d_out); T = `_output_seq_len()`.
        """
        if not frame_paths:
            return torch.zeros(1, self._output_seq_len(), self.d_out, device=self.device)

        sampled = _uniform_sample(list(frame_paths), self.n_frames)
        frames = [Image.open(fp).convert("RGB") for fp in sampled]
        return self._encode_sequence(frames)

    @torch.no_grad()
    def encode_batch(
        self, batch_webcam_crops: list[list[np.ndarray] | None]
    ) -> Tensor:
        """Batch encode multiple clips' webcam regions.

        Args:
            batch_webcam_crops: List of per-clip webcam crop lists (some may be None).

        Returns:
            Tensor of shape (B, T, d_out).
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
