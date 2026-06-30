"""Pose-kinematics context encoder (Phase 0 — new branch).

Replaces the ViT-ImageNet mean-pool context encoder when
``visual_encoder.context_encoder.type = "pose"`` in config.

Architecture:
  Per-frame keypoint extraction (MediaPipe Holistic or MMPose)
  → kinematic features (position + velocity + acceleration + confidence)
  → BiGRU temporal module
  → Linear(hidden*2, 768)
  → (B, 1, 768)   — same output shape as ViT-ImageNet branch

Key design decisions (P0.1 / P0.2):
  - Upper-body + hands landmarks only (shoulders, arms, wrists, hands, head).
  - Velocity and acceleration computed across the frame sequence (NOT mean-pooled
    static pose), capturing desk-slap impulses and rhythmic whole-body energy.
  - Confidence values kept as features — low/dropped confidence at high-arousal
    moments (jumping, lurching) is a signal, not noise.
  - Backend is abstracted: _MediaPipeBackend and _MMPoseBackend share the same
    (T, K, 4) output format (x, y, z/score, confidence).
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upper-body landmark indices for MediaPipe Pose (33 landmarks total).
# Selected: nose(0), shoulders(11,12), elbows(13,14), wrists(15,16),
#           hips(23,24) — gives 7 keypoints × 4 values = 28 dims.
# Hands: MediaPipe Hands gives 21 landmarks per hand (42 total) × 4 = 168 dims.
# Combined: 28 + 168 = 196-d per frame before kinematics.
# ---------------------------------------------------------------------------
_MP_POSE_UPPER_BODY_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # 9 landmarks


class _MediaPipeBackend:
    """Extract per-frame upper-body + hand keypoints using MediaPipe Holistic."""

    def __init__(self, use_hands: bool = True) -> None:
        self.use_hands = use_hands
        self._holistic = None

    def _get_holistic(self):
        if self._holistic is None:
            import mediapipe as mp
            if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "holistic"):
                raise RuntimeError(
                    "MediaPipe 'solutions.holistic' not found — it was removed in mediapipe>=0.10.14.\n"
                    "Fix: pip install 'mediapipe<0.10.14'\n"
                    "Or use pose_backend: 'mmpose' in config.yaml (requires mmpose installation)."
                )
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        return self._holistic

    def extract_keypoints(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return (T, K, 4) array — (x, y, z, visibility/confidence).

        K = 9 pose (upper body) + 21 left hand + 21 right hand = 51 keypoints.
        Missing landmarks are filled with zeros (confidence=0 signals absence).
        """
        import cv2

        holistic = self._get_holistic()
        n_pose = len(_MP_POSE_UPPER_BODY_IDX)
        n_hand = 21 if self.use_hands else 0
        K = n_pose + 2 * n_hand

        result_frames = []
        for frame_bgr in frames:
            kps = np.zeros((K, 4), dtype=np.float32)

            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)

                if res.pose_landmarks:
                    for out_i, src_i in enumerate(_MP_POSE_UPPER_BODY_IDX):
                        lm = res.pose_landmarks.landmark[src_i]
                        kps[out_i] = [lm.x, lm.y, lm.z, lm.visibility]

                if self.use_hands:
                    if res.left_hand_landmarks:
                        for j, lm in enumerate(res.left_hand_landmarks.landmark):
                            # confidence proxy: landmark presence confidence not available
                            # in Holistic — use 1.0 when detected, 0 when not
                            kps[n_pose + j] = [lm.x, lm.y, lm.z, 1.0]
                    if res.right_hand_landmarks:
                        for j, lm in enumerate(res.right_hand_landmarks.landmark):
                            kps[n_pose + n_hand + j] = [lm.x, lm.y, lm.z, 1.0]

            except Exception as e:
                logger.debug("MediaPipe frame failed: %s", e)

            result_frames.append(kps)

        return np.stack(result_frames, axis=0)  # (T, K, 4)

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None


class _MMPoseBackend:
    """Extract per-frame upper-body keypoints using MMPose (RTMPose).

    Requires: mmpose, mmcv, mmdet installed.
    Falls back gracefully to zeros if MMPose is not available.
    """

    _UPPER_BODY_COCO_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 13 kpts

    def __init__(self, device: str = "cpu") -> None:
        self._inferencer = None
        self._device = device

    def _get_inferencer(self):
        if self._inferencer is None:
            from mmpose.apis import MMPoseInferencer
            self._inferencer = MMPoseInferencer(
                pose2d="rtmpose-s_8xb256-420e_coco-256x192",
                device=self._device,
            )
        return self._inferencer

    def extract_keypoints(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return (T, K, 4) — K=13 COCO upper-body keypoints."""
        K = len(self._UPPER_BODY_COCO_IDX)
        result_frames = []

        try:
            inferencer = self._get_inferencer()
            for frame_bgr in frames:
                kps = np.zeros((K, 4), dtype=np.float32)
                try:
                    result = next(inferencer(frame_bgr, return_vis=False))
                    predictions = result.get("predictions", [])
                    if predictions and predictions[0]:
                        keypoints = np.array(predictions[0][0]["keypoints"])  # (17, 2)
                        scores = np.array(predictions[0][0]["keypoint_scores"])  # (17,)
                        for out_i, src_i in enumerate(self._UPPER_BODY_COCO_IDX):
                            if src_i < len(keypoints):
                                x, y = keypoints[src_i]
                                conf = float(scores[src_i])
                                kps[out_i] = [x, y, 0.0, conf]
                except Exception as e:
                    logger.debug("MMPose frame failed: %s", e)
                result_frames.append(kps)
        except ImportError:
            logger.warning("MMPose not installed — returning zero keypoints")
            result_frames = [np.zeros((K, 4), dtype=np.float32)] * len(frames)

        return np.stack(result_frames, axis=0)  # (T, K, 4)


def _normalize_kps_by_shoulder(kps: np.ndarray) -> np.ndarray:
    """Normalize (x, y) coordinates by median shoulder width (P0.0-fix.5).

    Makes motion-energy comparable across streamers at different camera distances.

    Shoulder output indices by backend:
      MediaPipe (K=51):  left_shoulder=1, right_shoulder=2
      MMPose COCO (K=13): left_shoulder=5, right_shoulder=6

    Falls back to no normalization if shoulder confidence is low (<3 valid frames).
    """
    K = kps.shape[1]
    s_l, s_r = (1, 2) if K > 13 else (5, 6)
    if s_l >= K or s_r >= K:
        return kps
    valid = (kps[:, s_l, 3] > 0.3) & (kps[:, s_r, 3] > 0.3)
    if valid.sum() < 3:
        return kps
    widths = np.linalg.norm(kps[valid, s_l, :2] - kps[valid, s_r, :2], axis=-1)
    scale = float(np.median(widths))
    if scale < 1e-6:
        return kps
    out = kps.copy()
    out[:, :, :2] /= scale
    return out


def _save_kps_cache(kps: np.ndarray, path: Path) -> None:
    """Save (T, K, 4) keypoints to .npy for sharing with MotionCueExtractor (P0.0-fix.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), kps)


def _load_kps_cache(path: Path) -> "np.ndarray | None":
    """Load cached (T, K, 4) keypoints; returns None if file is missing or corrupt."""
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception:
        return None


def _build_kinematic_features(kps: np.ndarray) -> np.ndarray:
    """Convert (T, K, 4) keypoints to (T, D) kinematic feature sequence.

    2D only — z intentionally dropped; 7 dims = pos(2) + vel(2) + acc(2) + conf(1).

    Features per frame:
      - positions (x, y) for each keypoint: K*2 dims
      - velocity (diff of positions, zero-padded at t=0): K*2 dims
      - acceleration (diff of velocity, zero-padded at t=0,1): K*2 dims
      - confidence (visibility) for each keypoint: K dims

    Velocity and acceleration are masked by the confidence product of adjacent
    frames: a keypoint that drops out then reappears would otherwise produce a
    large spurious spike.  Confidence itself is kept as-is because a sudden drop
    in visibility is a genuine high-arousal signal (jump, fast arm swing).

    Total D = K * (2 + 2 + 2 + 1) = K * 7.
    """
    T, K, _ = kps.shape
    pos = kps[:, :, :2]   # (T, K, 2) — 2D only, z dropped intentionally
    conf = kps[:, :, 3:4]  # (T, K, 1)

    vel = np.zeros_like(pos)
    raw_vel = pos[1:] - pos[:-1]
    vel_mask = conf[1:] * conf[:-1]   # (T-1, K, 1): zero when either endpoint uncertain
    vel[1:] = raw_vel * vel_mask

    acc = np.zeros_like(vel)
    raw_acc = vel[2:] - vel[1:-1]
    acc_mask = conf[2:] * conf[1:-1] * conf[:-2]   # (T-2, K, 1)
    acc[2:] = raw_acc * acc_mask

    # (T, K, 7) → (T, K*7)
    features = np.concatenate([pos, vel, acc, conf], axis=-1)
    return features.reshape(T, K * 7).astype(np.float32)


class _BiGRU(nn.Module):
    """Small bidirectional GRU temporal module."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, 768)

    def forward(self, x: Tensor) -> Tensor:
        """(B, T, input_dim) → (B, 768) using last hidden state."""
        _, h = self.gru(x)
        # h: (n_layers*2, B, hidden_dim) — take last layer, both directions
        h_fwd = h[-2]   # (B, hidden_dim)
        h_bwd = h[-1]   # (B, hidden_dim)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden_dim*2)
        return self.proj(h_cat)  # (B, 768)


class PoseContextEncoder(nn.Module):
    """Pose-kinematics context encoder for streamer body-language encoding.

    Encodes upper-body + hand motion across a clip into a 768-d vector that
    captures kinematics (velocity, acceleration, impact events, periodicity)
    rather than static appearance.

    Args:
        backend: "mediapipe" or "mmpose".
        n_frames: Number of frames to sample per clip.
        hidden_dim: BiGRU hidden size (output = hidden_dim*2 → Linear(768)).
        n_layers: Number of BiGRU layers.
        dropout: Dropout between GRU layers (ignored when n_layers=1).
        d_out: Output embedding dim (must be 768 to match fusion).
    """

    def __init__(
        self,
        backend: str = "mediapipe",
        n_frames: int = 16,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.3,
        d_out: int = 768,
    ) -> None:
        super().__init__()
        self.backend_name = backend
        self.n_frames = n_frames
        self.d_out = d_out

        if backend == "mediapipe":
            self._backend = _MediaPipeBackend(use_hands=True)
            n_pose_kpts = len(_MP_POSE_UPPER_BODY_IDX)  # 9
            n_hand_kpts = 42                              # 21 * 2 hands
            K = n_pose_kpts + n_hand_kpts               # 51
        elif backend == "mmpose":
            self._backend = _MMPoseBackend(device="cpu")  # device synced in encode() after .to()
            K = len(_MMPoseBackend._UPPER_BODY_COCO_IDX)  # 13
        else:
            raise ValueError(f"Unknown pose backend: {backend!r}. Use 'mediapipe' or 'mmpose'.")

        input_dim = K * 7  # pos(2) + vel(2) + acc(2) + conf(1) per keypoint
        self.temporal = _BiGRU(input_dim, hidden_dim, n_layers, dropout)

        logger.info(
            "PoseContextEncoder: backend=%s K=%d input_dim=%d hidden=%d",
            backend, K, input_dim, hidden_dim,
        )

    @torch.no_grad()
    def encode(
        self,
        webcam_crops: "list[np.ndarray] | None",
        kps_cache_path: "Path | None" = None,
    ) -> Tensor:
        """Encode a sequence of webcam frame crops.

        Args:
            webcam_crops: List of BGR ndarrays (full-resolution webcam region per frame,
                NOT resized to 224×224 — pose detection benefits from native resolution).
                If None or empty, returns zeros.
            kps_cache_path: If provided, the extracted (T, K, 4) keypoints are saved here
                after normalization so MotionCueExtractor can load them without re-running
                pose backend (P0.0-fix.2 shared cache).

        Returns:
            Tensor of shape (1, 1, 768).
        """
        device = next(self.parameters()).device

        if not webcam_crops:
            return torch.zeros(1, 1, self.d_out, device=device)

        # Sync MMPose backend to the module's current device (set after .to(device) call).
        # If device changed since init, reset the inferencer so it re-creates on correct device.
        if hasattr(self._backend, "_device"):
            dev_str = str(device)
            if self._backend._device != dev_str:
                self._backend._device = dev_str
                self._backend._inferencer = None  # force lazy re-init on new device

        sampled = _uniform_sample(webcam_crops, self.n_frames)
        kps = self._backend.extract_keypoints(sampled)   # (T, K, 4)
        kps = _normalize_kps_by_shoulder(kps)             # P0.0-fix.5: normalize by shoulder width
        if kps_cache_path is not None:                    # P0.0-fix.2: save for MotionCueExtractor
            _save_kps_cache(kps, Path(kps_cache_path))
        feats = _build_kinematic_features(kps)            # (T, K*7)

        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, D)
        out = self.temporal(x)  # (1, 768)
        return out.unsqueeze(1)  # (1, 1, 768)

    @torch.no_grad()
    def encode_from_kps(self, kps: np.ndarray) -> Tensor:
        """Encode pre-extracted (T, K, 4) keypoints without re-running pose backend.

        Used by MotionCueExtractor to avoid duplicate pose inference (P0.0-fix.2).
        kps must already be normalized (as saved by encode()).
        """
        device = next(self.parameters()).device
        feats = _build_kinematic_features(kps)
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        out = self.temporal(x)
        return out.unsqueeze(1)  # (1, 1, 768)

    @torch.no_grad()
    def encode_from_paths(self, frame_paths: list[Path]) -> Tensor:
        """Encode from file paths (fallback interface matching ViT branch)."""
        import cv2

        device = next(self.parameters()).device

        if not frame_paths:
            return torch.zeros(1, 1, self.d_out, device=device)

        sampled_paths = _uniform_sample(list(frame_paths), self.n_frames)
        frames = []
        for p in sampled_paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)

        if not frames:
            return torch.zeros(1, 1, self.d_out, device=device)

        return self.encode(frames)

    @torch.no_grad()
    def encode_batch(self, batch_webcam_crops: list["list[np.ndarray] | None"]) -> Tensor:
        """Batch encode multiple clips."""
        return torch.cat([self.encode(crops) for crops in batch_webcam_crops], dim=0)


def _uniform_sample(items: list, n: int) -> list:
    """Uniformly sample exactly n items from a list, padding with last if needed."""
    if not items:
        return items
    if len(items) >= n:
        indices = np.linspace(0, len(items) - 1, n, dtype=int)
        return [items[i] for i in indices]
    return list(items) + [items[-1]] * (n - len(items))
