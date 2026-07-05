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
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backend = backend
        self.n_frames = n_frames
        self.target_size = target_size
        self.temporal_pool = temporal_pool
        self.device = torch.device(device)

        logger.info("Loading context encoder (backend=%s): %s", backend, model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)

        self.d_out = self._probe_output_dim()
        logger.info("Context encoder loaded and frozen (d_out=%d)", self.d_out)

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

    @torch.no_grad()
    def encode(self, webcam_crops: list[np.ndarray] | None) -> Tensor:
        """Encode webcam region crops (wider than face crops).

        Args:
            webcam_crops: List of BGR ndarrays cropped to the webcam region.
                If None or empty, returns zeros (no-webcam mode).

        Returns:
            Tensor of shape (1, T, d_out):
                - T=1 if temporal_pool='mean'
                - T=n_frames if temporal_pool='none'
        """
        if not webcam_crops:
            T = 1 if self.temporal_pool == "mean" else self.n_frames
            return torch.zeros(1, T, self.d_out, device=self.device)

        sampled = _uniform_sample(webcam_crops, self.n_frames)
        cls_tokens = []
        for crop in sampled:
            image = _bgr_to_pil(crop)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls = self._extract_embedding(outputs)  # (1, d_out)
            cls_tokens.append(cls)

        stacked = torch.cat(cls_tokens, dim=0)  # (n_frames, d_out)

        if self.temporal_pool == "mean":
            return stacked.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, d_out)
        else:
            return stacked.unsqueeze(0)  # (1, n_frames, d_out)

    @torch.no_grad()
    def encode_from_paths(self, frame_paths: list[Path]) -> Tensor:
        """Encode full frames from file paths (fallback when no webcam detected).

        Args:
            frame_paths: Paths to extracted frame images.

        Returns:
            Tensor of shape (1, T, d_out).
        """
        if not frame_paths:
            T = 1 if self.temporal_pool == "mean" else self.n_frames
            return torch.zeros(1, T, self.d_out, device=self.device)

        sampled = _uniform_sample(list(frame_paths), self.n_frames)
        cls_tokens = []
        for fp in sampled:
            image = Image.open(fp).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls = self._extract_embedding(outputs)
            cls_tokens.append(cls)

        stacked = torch.cat(cls_tokens, dim=0)

        if self.temporal_pool == "mean":
            return stacked.mean(dim=0, keepdim=True).unsqueeze(0)
        else:
            return stacked.unsqueeze(0)

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
