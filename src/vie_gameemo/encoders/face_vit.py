"""Face encoder using ViT-FER (Path 1 of dual-path visual).

Encodes the streamer's face (cropped from webcam region) using a ViT-Base
FER (Facial Expression Recognition) fine-tune. Uses a tri-view design:
    - Spatial view: peak frame PATCH tokens (pooled) for micro-expression detail
    - Global view: peak frame CLS token
    - Temporal view: 16 evenly-sampled frames, each CLS token kept

Output layout: [peak_patches | global_CLS | temporal_CLS]

IMPORTANT: This encoder receives FACE CROPS from face_crop.py, not full
frames. Do not feed full frames into ViT-FER (distribution mismatch with
the FER pretrain data).

Two checkpoints are supported via ``backend`` (both ViT-Base, 768d — see
``visual_encoder.face_encoder.models`` in config.yaml):
    - "vit"       (default): trpakov/vit-face-expression, fine-tuned on FER2013 only.
    - "vit_multi": mo-thecreator/vit-Facial-Expression-Recognition, fine-tuned
      on FER2013 + MMI + AffectNet (broader data, ~84% self-reported acc).
Loaded via AutoModel/AutoImageProcessor so either (or any other ViT-Base FER
checkpoint) can be swapped in purely via config.
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


class FaceEncoder(nn.Module):
    """Tri-view face encoder using ViT-FER (Path 1 of dual-path).

    Output: (B, n_patch_tokens + 1 + n_temporal_frames, 768)
    where n_patch_tokens = spatial_pool[0] * spatial_pool[1].
    """

    def __init__(
        self,
        model_name: str = "trpakov/vit-face-expression",
        backend: str = "vit",
        n_temporal_frames: int = 16,
        spatial_pool: tuple[int, int] = (4, 4),
        target_size: tuple[int, int] = (224, 224),
        peak_frame_source: str = "auto_peak",
        peak_signal: str = "visual",
        filter_invalid_frames: bool = True,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize face encoder.

        Args:
            model_name: HF model ID (ViT-FER fine-tune).
            backend: Which checkpoint family this is ("vit" | "vit_multi") —
                purely informational/for logging here since both are loaded
                identically via AutoModel; kept for parity with context_vit's
                backend param and future non-ViT-Base additions.
            n_temporal_frames: Frames sampled for temporal view.
            spatial_pool: (H, W) pooling on peak frame patch grid.
            target_size: Input frame resize target (W, H).
            peak_frame_source: How the peak frame (spatial-patch view) and
                global-CLS view are picked — see `visual_encoder.face_encoder.
                dual_view.global.source` in config.yaml:
                  - "auto_peak" (default): peak frame chosen by
                    `_select_peak_frame` (most different from the clip's mean
                    appearance); global CLS = mean of all temporal CLS tokens.
                  - "middle_frame" (legacy, pre-fef0294 behavior): peak frame
                    = fixed middle-of-clip index; global CLS = that same
                    frame's own CLS token. Changing this value changes the
                    cached "face" feature's content (not just its shape), so
                    a checkpoint trained under one setting should be
                    evaluated with the SAME setting, not the other.
            peak_signal: Only relevant when peak_frame_source="auto_peak"
                (ignored, no-op, under "middle_frame"). Which signal(s)
                `_select_peak_frame` uses to pick the peak frame:
                  - "visual" (default): visual distance from the clip's mean
                    appearance only (original fef0294 behavior).
                  - "audio_visual": z-scored visual distance + z-scored
                    per-frame audio RMS energy (see `audio_energy` arg of
                    `encode`) — catches brief "startle" reactions (e.g. a
                    quick shocked/gasp) whose peak coincides with a vocal
                    spike more than with a visually-different frame.
            filter_invalid_frames: See `visual_encoder.face_encoder.dual_view.
                global.filter_invalid_frames` in config.yaml:
                  - True (default, behavior since commit bd50d9c): drop frames
                    where `valid_mask` (passed to `encode`) is False — i.e.
                    frames where the crop pipeline fell back to a wider,
                    non-face-only region — before peak selection and temporal
                    sampling, so a bad frame can't inject non-face signal.
                  - False (pre-bd50d9c / commit 95ab48e behavior): ignore
                    `valid_mask` entirely and use every frame as-is, including
                    fallback crops. Set this to reproduce features/checkpoints
                    from before valid_mask filtering was introduced. Like
                    `peak_frame_source`, this changes cached "face" feature
                    CONTENT, not just shape (see `_config_hash` in
                    extract_features.py, which hashes this too).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.backend = backend
        self.n_temporal_frames = n_temporal_frames
        self.spatial_pool = spatial_pool
        self.target_size = target_size
        self.filter_invalid_frames = filter_invalid_frames
        if peak_frame_source not in ("auto_peak", "middle_frame"):
            raise ValueError(
                f"Unknown peak_frame_source: {peak_frame_source!r}. "
                "Use 'auto_peak' or 'middle_frame'."
            )
        self.peak_frame_source = peak_frame_source
        if peak_signal not in ("visual", "audio_visual"):
            raise ValueError(
                f"Unknown peak_signal: {peak_signal!r}. Use 'visual' or 'audio_visual'."
            )
        self.peak_signal = peak_signal
        self.device = torch.device(device)

        logger.info("Loading ViT-FER (backend=%s): %s", backend, model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
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
        valid_mask: list[bool] | None = None,
        audio_energy: np.ndarray | None = None,
    ) -> tuple[Tensor, bool]:
        """Encode a sequence of face crops with tri-view design.

        Args:
            face_crops: List of face crop ndarrays (BGR, HxWx3).
                If None or empty, returns zeros (no-facecam mode).
            valid_mask: Optional per-frame bool list, True where the frame is
                a genuine tight face crop and False where the crop pipeline
                fell back to a wider, non-face-only region (see
                `face_crop.extract_streamer_face`). When given and at least
                one frame is valid, only valid frames are used for peak
                selection and temporal sampling — otherwise a bad frame
                (background/hands/desk) can silently sit in the same
                sequence as real face frames and inject non-face signal.
                If None, or all frames are invalid, all frames are used
                (matches previous behavior). Also ignored entirely (all
                frames used as-is) when `self.filter_invalid_frames` is
                False — see that flag's docstring above.
            audio_energy: Optional 1D array, same length as `face_crops`
                (BEFORE valid_mask filtering — filtered internally the same
                way), giving a per-frame audio energy value (e.g. short-time
                RMS uniformly resampled across the clip). Only used when
                `self.peak_signal == "audio_visual"` (ignored otherwise, and
                a no-op under `peak_frame_source="middle_frame"` since that
                mode never calls `_select_peak_frame`). See
                `feature_extraction._compute_frame_audio_energy`.

        Returns:
            Tuple of:
                - Tensor (1, T, 768) where T = n_patch + 1 + n_temporal:
                  [peak_patches | global_CLS | temporal_CLS]
                - has_face: True if real face data, False for zeros placeholder
        """
        if not face_crops:
            return torch.zeros(1, self.total_tokens, 768, device=self.device), False

        usable_crops = face_crops
        usable_energy = audio_energy
        if (
            self.filter_invalid_frames
            and valid_mask is not None
            and len(valid_mask) == len(face_crops)
            and any(valid_mask)
        ):
            usable_crops = [c for c, ok in zip(face_crops, valid_mask) if ok]
            if usable_energy is not None and len(usable_energy) == len(valid_mask):
                mask_arr = np.asarray(valid_mask, dtype=bool)
                usable_energy = np.asarray(usable_energy)[mask_arr]

        frames_pil = [_bgr_to_pil(f) for f in usable_crops]

        # Peak frame: source depends on self.peak_frame_source (see __init__
        # docstring). "auto_peak" picks the frame most different from the
        # clip's mean appearance (no ground-truth peak label exists — see
        # peak_frame_idx in Annotation, which is a hardcoded 0 for imported
        # labels, not a real detected peak), optionally combined with audio
        # energy when peak_signal="audio_visual" (see _select_peak_frame).
        # "middle_frame" reproduces the pre-fef0294 behavior (fixed
        # middle-of-clip index) for compatibility with checkpoints trained
        # before that change. Used for the fine-grained spatial patch view
        # below — a single frame is still needed for spatial detail, and
        # computing full patch-level hidden states for every temporal frame
        # would be far more expensive for uncertain benefit.
        if self.peak_frame_source == "middle_frame":
            peak_idx = len(frames_pil) // 2
        else:
            peak_idx = _select_peak_frame(
                frames_pil,
                audio_energy=usable_energy if self.peak_signal == "audio_visual" else None,
            )
        peak_cls, peak_patches = self._encode_full_frame(frames_pil[peak_idx])

        # Spatial pool patch tokens for detail
        pooled_patches = self._spatial_pool_patches(
            peak_patches, self.spatial_pool[0], self.spatial_pool[1],
        )  # (1, n_patch, 768)

        # Temporal view: n_temporal_frames uniformly sampled
        sampled = _uniform_sample(frames_pil, self.n_temporal_frames)
        temporal_cls = torch.cat(
            [self._encode_single_frame(f) for f in sampled], dim=0
        )  # (n_temporal_frames, 768)

        # Global view: "middle_frame" reuses the peak frame's own CLS token
        # (legacy behavior — peak_idx is the middle frame in this mode, so
        # this matches pre-fef0294 exactly). "auto_peak" instead means over
        # the temporal CLS tokens, reducing reliance on any one (possibly
        # mis-selected) frame for the coarse "global" signal — mirrors
        # Emotion-LLaMA-v2's adaptive pooling over uniformly-sampled frames.
        if self.peak_frame_source == "middle_frame":
            global_cls = peak_cls  # (1, 768)
        else:
            global_cls = temporal_cls.mean(dim=0, keepdim=True)  # (1, 768)
        temporal_cls = temporal_cls.unsqueeze(0)  # (1, n_temporal, 768)

        # [peak_patches | global_CLS | temporal_CLS]
        combined = torch.cat([
            pooled_patches,
            global_cls.unsqueeze(0),
            temporal_cls,
        ], dim=1)
        return combined, True

    @torch.no_grad()
    def encode_batch(
        self,
        batch_face_crops: list[list[np.ndarray] | None],
        batch_valid_mask: list[list[bool] | None] | None = None,
    ) -> tuple[Tensor, list[bool]]:
        """Batch encode multiple clips.

        Args:
            batch_face_crops: List of per-clip face crop lists (some may be None).
            batch_valid_mask: Optional list of per-clip valid_mask lists (see
                `encode`), aligned with `batch_face_crops`.

        Returns:
            Tuple of (stacked tensor (B, T, 768), list of has_face flags).
        """
        tensors = []
        flags = []
        masks = batch_valid_mask if batch_valid_mask is not None else [None] * len(batch_face_crops)
        for crops, mask in zip(batch_face_crops, masks):
            t, flag = self.encode(crops, valid_mask=mask)
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


def _select_peak_frame(frames_pil: list, audio_energy: np.ndarray | None = None) -> int:
    """Pick the frame that deviates most from the clip's mean appearance,
    optionally combined with per-frame audio energy.

    No reliable ground-truth peak-emotion frame exists for this dataset
    (Annotation.peak_frame_idx is hardcoded to 0 by the flat-label importer,
    not a real detected peak), so this heuristic stands in for it: downscale
    each frame to a small grayscale thumbnail, compute the mean thumbnail
    across the clip, and return the index with the largest L2 distance from
    that mean. A frame far from the clip's "resting" average is a reasonable
    proxy for the most expressive moment, and is at least data-driven rather
    than an arbitrary fixed index.

    A visual-only signal misses brief "startle" reactions (a quick gasp/
    scream) where the most visually-different frame (e.g. a hand suddenly
    covering the face) isn't the most emotionally-informative one. When
    `audio_energy` is given, it's combined with the visual distance
    (z-scored, equal weight, summed) so a frame that's both visually
    distinctive AND coincides with a vocal spike is preferred.

    Args:
        frames_pil: Non-empty list of PIL Images (already filtered to valid
            face crops by the caller).
        audio_energy: Optional 1D array of length len(frames_pil) — per-frame
            audio energy (e.g. short-time RMS resampled to one value per
            frame). If None, or its length doesn't match frames_pil, falls
            back to visual-only selection.

    Returns:
        Index into `frames_pil` of the selected peak frame.
    """
    if len(frames_pil) == 1:
        return 0

    thumbs = np.stack([
        np.asarray(f.convert("L").resize((48, 48)), dtype=np.float32)
        for f in frames_pil
    ])  # (N, 48, 48)
    mean_thumb = thumbs.mean(axis=0, keepdims=True)
    visual_dist = np.linalg.norm((thumbs - mean_thumb).reshape(len(frames_pil), -1), axis=1)

    if audio_energy is None or len(audio_energy) != len(frames_pil):
        return int(np.argmax(visual_dist))

    def _zscore(x: np.ndarray) -> np.ndarray:
        std = float(x.std())
        return (x - x.mean()) / std if std > 1e-8 else np.zeros_like(x)

    combined = _zscore(visual_dist) + _zscore(np.asarray(audio_energy, dtype=np.float32))
    return int(np.argmax(combined))


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
