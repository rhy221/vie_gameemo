"""CueExtractor: label-orthogonal multimodal cue extraction.

Extracts fine-grained, non-emotional descriptors from (up to) 4 modalities:
  - Face: MediaPipe FaceMesh geometric ratios (EAR, MAR, brow height, head pose)
  - Voice: librosa prosody (pitch, energy, speaking rate)
  - Motion: pose kinematics (only when context_encoder_type="pose")
  - Text: regex-based lexicon hits (exclamations, negative words, game terms)

**Context cue is gated by context_encoder_type (FIX 1):**
  - "pose"        → MotionCueExtractor: kinematic cues from precomputed pose cache
                    (motion energy, wrist impact, whole-body periodicity)
  - "vit_imagenet" → no context cue (ViT-ImageNet is motion-blind; producing
                    action cues from it would be fabricated — FIX 1 rule)

**Attribute vector layout (FIX 2):**
  - "pose"        → face(5) + voice(3) + motion(3) + text(4) = 15-d
  - "vit_imagenet" → face(5) + voice(3) + text(4) = 12-d

NOTE: brightness/color/edge stats removed entirely (FIX 1 — they are static
image statistics with no relationship to streamer action, not valid context cues).

Usage:
    extractor = CueExtractor(cache_dir="data/cache", context_encoder_type="pose")
    extractor.precompute_all(faces_dir, audios_dir, frames_dir)
    cue_text, attr_vec = extractor.extract(clip_id, transcript)
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_AU_NAMES = {
    "1": "inner_brow_raise", "2": "outer_brow_raise", "4": "brow_lower",
    "5": "upper_lid_raise", "6": "cheek_raise", "7": "lid_tighten",
    "12": "lip_corner_pull", "15": "lip_corner_depress", "17": "chin_raise",
    "20": "lip_stretch", "23": "lip_tighten", "25": "lips_part",
    "26": "jaw_drop", "45": "blink",
}

_PITCH_BINS = [(150, "low"), (250, "mid"), (float("inf"), "high")]
_ENERGY_BINS = [(0.02, "quiet"), (0.06, "moderate"), (float("inf"), "loud")]
_RATE_BINS = [(2.0, "slow"), (4.0, "normal"), (float("inf"), "fast")]

# Motion kinematic bins (for pose branch)
_MOTION_ENERGY_BINS = [(0.02, "low"), (0.08, "moderate"), (float("inf"), "high")]
_IMPACT_BINS = [(0.05, "none"), (0.15, "moderate"), (float("inf"), "strong")]
_PERIOD_BINS = [(0.2, "none"), (0.5, "weak"), (float("inf"), "rhythmic")]

_EXCLAMATION_RE = re.compile(r"[!?]{1,}")
_NEGATIVE_VN = {
    "chết", "thua", "lỗi", "sai", "hỏng", "mất", "chán", "dở", "tệ",
    "ngu", "nát", "đau", "khó", "lag", "bug", "rip", "gg", "dead",
    "die", "kill", "loss", "fail", "damn", "shit", "fuck", "wtf",
}
_GAME_TERMS = {
    "ace", "kill", "headshot", "combo", "ulti", "skill", "buff", "nerf",
    "gank", "push", "farm", "tower", "baron", "dragon", "pentakill",
    "clutch", "gg", "wp", "ez", "ff", "carry", "feed", "noob",
    "rank", "match", "round", "team", "player", "camp", "rush",
}

_EMOTION_BLOCKLIST = {
    "happy", "sad", "angry", "fear", "surprise", "disgust", "joy",
    "vui", "buồn", "giận", "sợ", "ngạc nhiên", "ghê", "phấn khích",
    "hype", "tilted", "amused", "shocked", "neutral", "disgusted",
}


def _bin_value(val: float, bins: list[tuple[float, str]]) -> str:
    for threshold, label in bins:
        if val < threshold:
            return label
    return bins[-1][1]


# ---------------------------------------------------------------------------
# Motion cue extractor (pose branch only)
# ---------------------------------------------------------------------------

class MotionCueExtractor:
    """Reads precomputed pose kinematics from cache → discretized motion cues.

    Cache files (JSON per clip) must be precomputed by calling
    ``precompute(clip_id, frames_dir)`` before training.

    Cue categories:
      - "motion — sharp downward arm impact"     (desk-slap: high impact)
      - "motion — high whole-body energy, rhythmic" (jump/hype: high energy + periodic)
      - "motion — elevated body movement"        (moderate energy)
      - "no_motion_data"                         (no cache / extraction failed)
    """

    def __init__(
        self,
        cache_dir: str | Path,
        motion_energy_bins=None,
        impact_bins=None,
        period_bins=None,
        impact_conf_threshold: float = 0.3,
    ) -> None:
        self.kinematics_dir = Path(cache_dir) / "pose_kinematics"
        self._motion_energy_bins = motion_energy_bins if motion_energy_bins is not None else _MOTION_ENERGY_BINS
        self._impact_bins = impact_bins if impact_bins is not None else _IMPACT_BINS
        self._period_bins = period_bins if period_bins is not None else _PERIOD_BINS
        self._impact_conf_threshold = impact_conf_threshold

    def precompute(self, clip_id: str, frames_dir: Path, backend: str = "mediapipe") -> None:
        """Extract and cache kinematic summary for one clip.

        Called once per clip before training. Tries to load keypoints from the
        shared .npy cache written by PoseContextEncoder.encode() (P0.0-fix.2 —
        avoids running pose backend twice). Falls back to running pose directly if
        the cache is absent.
        """
        out = self.kinematics_dir / f"{clip_id}.json"
        if out.exists():
            return

        self.kinematics_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2

            from vie_gameemo.encoders.context_pose import (
                _MediaPipeBackend, _MMPoseBackend, _build_kinematic_features,
                _load_kps_cache, _normalize_kps_by_shoulder, _uniform_sample,
            )

            # Try shared kps cache written by PoseContextEncoder (P0.0-fix.2)
            kps_cache_path = self.kinematics_dir / f"{clip_id}_kps.npy"
            kps = _load_kps_cache(kps_cache_path)

            if kps is None:
                # Cache miss — run pose backend ourselves
                frame_dir = frames_dir / clip_id if (frames_dir / clip_id).is_dir() else frames_dir
                frames_paths = (
                    sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
                    if frame_dir.is_dir() else []
                )
                if not frames_paths:
                    _save_json(out, _empty_kinematics())
                    return

                sample_paths = _uniform_sample(frames_paths, 16)
                frames = [cv2.imread(str(p)) for p in sample_paths]
                frames = [f for f in frames if f is not None]
                if not frames:
                    _save_json(out, _empty_kinematics())
                    return

                extractor = _MMPoseBackend() if backend == "mmpose" else _MediaPipeBackend(use_hands=True)
                kps = extractor.extract_keypoints(frames)  # (T, K, 4)
                kps = _normalize_kps_by_shoulder(kps)      # P0.0-fix.5: normalize by shoulder width

            feats = _build_kinematic_features(kps)  # (T, K*7)

            T, D = feats.shape
            K = D // 7
            vel = feats[:, K * 2: K * 4].reshape(T, K, 2)
            acc = feats[:, K * 4: K * 6].reshape(T, K, 2)
            conf = feats[:, K * 6:].reshape(T, K)  # (T, K)

            motion_energy = float(np.mean(np.sum(vel ** 2, axis=(1, 2))))

            # Body anchor: head + shoulders are always visible in facecam crops
            # (P0.0-fix.4: wrists leave frame on tight facecam crops → inconsistent
            #  across streamers; shoulders/nose are reliable anchor points).
            # MediaPipe (K>13): nose=0, left_shoulder=1, right_shoulder=2
            # MMPose COCO (K≤13): nose=0, left_shoulder=5, right_shoulder=6
            # Binary gate — intentionally stricter than the soft mask in _build_kinematic_features:
            # features use continuous confidence weighting (conf[t]*conf[t-1]*...) which only
            # attenuates derivatives; cues use a hard 0/1 gate so the impact label is either
            # present or absent, never a "0.06× real" artifact. Both check the same 3-frame window.
            # acc[t] = pos[t] − 2·pos[t−1] + pos[t−2] → needs conf[t], conf[t-1], conf[t-2] all valid.
            if K > 13:
                body_anchor_idx = [i for i in [0, 1, 2] if i < K]
            else:
                body_anchor_idx = [i for i in [0, 5, 6] if i < K]
            if not body_anchor_idx:
                body_anchor_idx = list(range(min(3, K)))

            anchor_conf = conf[:, body_anchor_idx]  # (T, n_anchors)
            min_conf_3frame = np.zeros_like(anchor_conf)
            min_conf_3frame[2:] = np.minimum(
                np.minimum(anchor_conf[2:], anchor_conf[1:-1]), anchor_conf[:-2]
            )
            conf_gate = (min_conf_3frame >= self._impact_conf_threshold).astype(float)
            anchor_acc_down = acc[:, body_anchor_idx, 1] * conf_gate
            impact = float(np.max(np.abs(anchor_acc_down))) if anchor_acc_down.size > 0 else 0.0

            # Periodicity: autocorrelation of per-frame motion energy signal
            per_frame_energy = np.sum(vel ** 2, axis=(1, 2))  # (T,)
            periodicity = 0.0
            if T >= 4:
                norm = per_frame_energy - per_frame_energy.mean()
                autocorr = np.correlate(norm, norm, mode="full")
                mid = len(autocorr) // 2
                lags = autocorr[mid + 1: mid + T // 2]
                if autocorr[mid] > 1e-9:
                    periodicity = float(np.max(lags) / autocorr[mid])

            _save_json(out, {
                "motion_energy": motion_energy,
                "impact": impact,
                "periodicity": periodicity,
            })

        except Exception as e:
            logger.debug("Motion kinematic precompute failed for %s: %s", clip_id, e)
            _save_json(out, _empty_kinematics())

    def extract(self, clip_id: str) -> tuple[str, list[float]]:
        """Load precomputed kinematics → discretized cue text + 3-d attr vector."""
        data = _load_json(self.kinematics_dir / f"{clip_id}.json")
        if data is None:
            return "no_motion_data", [0.0, 0.0, 0.0]

        energy = data.get("motion_energy", 0.0)
        impact = data.get("impact", 0.0)
        periodicity = data.get("periodicity", 0.0)

        energy_label = _bin_value(energy, self._motion_energy_bins)
        impact_label = _bin_value(impact, self._impact_bins)
        period_label = _bin_value(periodicity, self._period_bins)

        # Compose named cue (per prompt spec)
        if impact_label == "strong":
            cue = "motion — sharp downward arm impact"
        elif energy_label == "high" and period_label == "rhythmic":
            cue = "motion — high whole-body energy, rhythmic"
        elif energy_label == "high":
            cue = "motion — elevated body movement"
        elif energy_label == "low" and impact_label == "none":
            cue = "motion — minimal movement"
        else:
            cue = f"motion — energy={energy_label}, impact={impact_label}"

        # Normalised attr vector (0–1 range)
        attr = [
            min(energy / 0.15, 1.0),
            min(impact / 0.3, 1.0),
            min(periodicity, 1.0),
        ]
        return cue, attr


def _empty_kinematics() -> dict:
    return {"motion_energy": 0.0, "impact": 0.0, "periodicity": 0.0}


# ---------------------------------------------------------------------------
# Main CueExtractor
# ---------------------------------------------------------------------------

class CueExtractor:
    """Extracts label-orthogonal cues from face, voice, (motion,) and text.

    Precompute once per dataset split, then read from cache during training.

    Args:
        cache_dir: Directory for precomputed cache files.
        context_encoder_type: "pose" or "vit_imagenet".
            Controls whether motion cues are generated and whether
            the attr vector is 15-d (pose) or 12-d (vit_imagenet).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        context_encoder_type: str = "vit_imagenet",
        cues_cfg=None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.context_encoder_type = context_encoder_type
        self.face_geo_dir = self.cache_dir / "face_geo"
        self.prosody_dir = self.cache_dir / "prosody"

        # Allow callers to override bin thresholds via config; fall back to module constants.
        if cues_cfg is not None:
            def _parse_bins(raw):
                return [(float(t), str(lbl)) for t, lbl in raw] + [(float("inf"), str(raw[-1][1]))]
            raw_e = getattr(cues_cfg, "motion_energy_bins", None)
            raw_i = getattr(cues_cfg, "impact_bins", None)
            raw_p = getattr(cues_cfg, "periodicity_bins", None)
            self._motion_energy_bins = _parse_bins(raw_e) if raw_e is not None else _MOTION_ENERGY_BINS
            self._impact_bins = _parse_bins(raw_i) if raw_i is not None else _IMPACT_BINS
            self._period_bins = _parse_bins(raw_p) if raw_p is not None else _PERIOD_BINS
        else:
            self._motion_energy_bins = _MOTION_ENERGY_BINS
            self._impact_bins = _IMPACT_BINS
            self._period_bins = _PERIOD_BINS

        self.motion_extractor: MotionCueExtractor | None = None
        if context_encoder_type == "pose":
            impact_conf_threshold = (
                getattr(cues_cfg, "impact_conf_threshold", 0.3)
                if cues_cfg is not None else 0.3
            )
            self.motion_extractor = MotionCueExtractor(
                self.cache_dir,
                motion_energy_bins=self._motion_energy_bins,
                impact_bins=self._impact_bins,
                period_bins=self._period_bins,
                impact_conf_threshold=impact_conf_threshold,
            )

    @property
    def n_attrs(self) -> int:
        """Attribute vector dimension (depends on context_encoder_type)."""
        return 15 if self.context_encoder_type == "pose" else 12

    # ------------------------------------------------------------------
    # Precompute (run once before training)
    # ------------------------------------------------------------------

    def precompute_all(
        self,
        faces_dir: str | Path,
        audios_dir: str | Path,
        frames_dir: str | Path,
        clip_ids: list[str] | None = None,
        pose_backend: str = "mediapipe",
    ) -> None:
        """Batch-extract face geometry, prosody, and (if pose branch) motion for all clips."""
        faces_dir = Path(faces_dir)
        audios_dir = Path(audios_dir)
        frames_dir = Path(frames_dir)

        for d in (self.face_geo_dir, self.prosody_dir):
            d.mkdir(parents=True, exist_ok=True)

        if clip_ids is None:
            clip_ids = sorted({p.stem for p in audios_dir.glob("*.wav")})

        logger.info("Precomputing cues for %d clips (context=%s)", len(clip_ids), self.context_encoder_type)

        for cid in clip_ids:
            self._precompute_face_geo(cid, faces_dir)
            self._precompute_prosody(cid, audios_dir)
            if self.motion_extractor is not None:
                self.motion_extractor.precompute(cid, frames_dir, backend=pose_backend)

        logger.info("Cue precompute done")

    def _precompute_face_geo(self, clip_id: str, faces_dir: Path) -> None:
        out = self.face_geo_dir / f"{clip_id}.json"
        if out.exists():
            return

        face_dir = faces_dir / clip_id
        if not face_dir.exists():
            _save_json(out, {"has_face": False, "ear": 0, "mar": 0, "brow_h": 0, "yaw": 0, "pitch": 0})
            return

        frames = sorted(face_dir.glob("*.jpg")) + sorted(face_dir.glob("*.png"))
        if not frames:
            _save_json(out, {"has_face": False, "ear": 0, "mar": 0, "brow_h": 0, "yaw": 0, "pitch": 0})
            return

        try:
            import cv2
            import mediapipe as mp

            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.3,
            )

            ear_vals, mar_vals, brow_vals, yaw_vals, pitch_vals = [], [], [], [], []

            mid_frame = frames[len(frames) // 2]
            sample_frames = [mid_frame]
            if len(frames) > 4:
                sample_frames = [frames[i] for i in range(0, len(frames), max(1, len(frames) // 4))][:5]

            for fp in sample_frames:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    continue

                lm = results.multi_face_landmarks[0].landmark
                ear_vals.append(_compute_ear(lm))
                mar_vals.append(_compute_mar(lm))
                brow_vals.append(_compute_brow_height(lm))
                y, p = _compute_head_pose(lm)
                yaw_vals.append(y)
                pitch_vals.append(p)

            face_mesh.close()

            if ear_vals:
                _save_json(out, {
                    "has_face": True,
                    "ear": float(np.mean(ear_vals)),
                    "mar": float(np.mean(mar_vals)),
                    "brow_h": float(np.mean(brow_vals)),
                    "yaw": float(np.mean(yaw_vals)),
                    "pitch": float(np.mean(pitch_vals)),
                })
            else:
                _save_json(out, {"has_face": False, "ear": 0, "mar": 0, "brow_h": 0, "yaw": 0, "pitch": 0})
        except Exception as e:
            logger.debug("Face geo failed for %s: %s", clip_id, e)
            _save_json(out, {"has_face": False, "ear": 0, "mar": 0, "brow_h": 0, "yaw": 0, "pitch": 0})

    def _precompute_prosody(self, clip_id: str, audios_dir: Path) -> None:
        out = self.prosody_dir / f"{clip_id}.json"
        if out.exists():
            return

        wav_path = audios_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            _save_json(out, {"f0": 0, "rms": 0, "rate": 0})
            return

        try:
            import librosa

            y, sr = librosa.load(str(wav_path), sr=16000)
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            f0_valid = f0[~np.isnan(f0)]
            mean_f0 = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0

            rms = librosa.feature.rms(y=y)[0]
            mean_rms = float(np.mean(rms))

            duration = len(y) / sr
            _save_json(out, {"f0": mean_f0, "rms": mean_rms, "rate": 0.0, "duration": duration})
        except Exception as e:
            logger.debug("Prosody failed for %s: %s", clip_id, e)
            _save_json(out, {"f0": 0, "rms": 0, "rate": 0, "duration": 5.0})

    # ------------------------------------------------------------------
    # Extract (called per sample during training)
    # ------------------------------------------------------------------

    def extract(
        self,
        clip_id: str,
        transcript: str = "",
        has_face: bool = True,
    ) -> tuple[str, torch.Tensor]:
        """Build cue text + attribute vector from precomputed cache.

        Returns:
            (cue_text, attr_vector)
            attr_vector is 15-d for pose branch, 12-d for vit_imagenet branch.
        """
        face_cue, face_attrs = self._face_cue(clip_id, has_face)      # 5-d
        voice_cue, voice_attrs = self._voice_cue(clip_id, transcript)  # 3-d
        text_cue, text_attrs = self._text_cue(transcript)              # 4-d

        if self.context_encoder_type == "pose" and self.motion_extractor is not None:
            motion_cue, motion_attrs = self.motion_extractor.extract(clip_id)  # 3-d
            cue_text = (
                f"face: {face_cue}; voice: {voice_cue}; "
                f"motion: {motion_cue}; text: {text_cue}"
            )
            all_attrs = face_attrs + voice_attrs + motion_attrs + text_attrs  # 15-d
        else:
            # vit_imagenet branch: no context cue (feature is motion-blind)
            cue_text = f"face: {face_cue}; voice: {voice_cue}; text: {text_cue}"
            all_attrs = face_attrs + voice_attrs + text_attrs  # 12-d

        return cue_text, torch.tensor(all_attrs, dtype=torch.float32)

    def _face_cue(self, clip_id: str, has_face: bool) -> tuple[str, list[float]]:
        data = _load_json(self.face_geo_dir / f"{clip_id}.json")
        if data is None or not data.get("has_face", False) or not has_face:
            return "no_webcam", [0.0] * 5

        ear = data["ear"]
        mar = data["mar"]
        brow_h = data["brow_h"]
        yaw = data["yaw"]
        pitch = data["pitch"]

        parts = []
        if ear > 0.25:
            parts.append("eyes=wide")
        elif ear < 0.18:
            parts.append("eyes=narrow")

        if mar > 0.4:
            parts.append("mouth=open")
        elif mar < 0.15:
            parts.append("mouth=closed")

        if brow_h > 0.06:
            parts.append("brows=raised")
        elif brow_h < 0.03:
            parts.append("brows=lowered")

        if abs(yaw) > 15:
            parts.append(f"head_turn={'left' if yaw > 0 else 'right'}")
        if abs(pitch) > 10:
            parts.append(f"head_tilt={'up' if pitch > 0 else 'down'}")

        if not parts:
            parts.append("neutral_pose")

        return ", ".join(parts), [ear, mar, brow_h, yaw / 45.0, pitch / 45.0]

    def _voice_cue(self, clip_id: str, transcript: str) -> tuple[str, list[float]]:
        data = _load_json(self.prosody_dir / f"{clip_id}.json")
        if data is None:
            return "no_audio", [0.0, 0.0, 0.0]

        f0 = data.get("f0", 0)
        rms = data.get("rms", 0)
        duration = data.get("duration", 5.0)

        words = transcript.split() if transcript else []
        rate = len(words) / max(duration, 0.1)

        pitch_label = _bin_value(f0, _PITCH_BINS)
        energy_label = _bin_value(rms, _ENERGY_BINS)
        rate_label = _bin_value(rate, _RATE_BINS)

        cue = f"pitch={pitch_label}, energy={energy_label}, rate={rate_label}"
        return cue, [f0 / 300.0, rms / 0.1, rate / 5.0]

    def _text_cue(self, transcript: str) -> tuple[str, list[float]]:
        if not transcript or not transcript.strip():
            return "no_speech", [0.0, 0.0, 0.0, 0.0]

        lower = transcript.lower()
        words = lower.split()
        n_words = len(words)

        n_exclaim = len(_EXCLAMATION_RE.findall(transcript))
        n_negative = sum(1 for w in words if w in _NEGATIVE_VN)
        n_game = sum(1 for w in words if w in _GAME_TERMS)

        parts = []
        if n_exclaim > 0:
            parts.append("exclamation")
        if n_negative > 0:
            parts.append("negative_lexicon")
        if n_game > 0:
            parts.append("game_term")
        parts.append(f"{n_words}_words")

        return ", ".join(parts), [
            min(n_exclaim, 5) / 5.0,
            min(n_negative, 5) / 5.0,
            min(n_game, 5) / 5.0,
            min(n_words, 30) / 30.0,
        ]


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark helpers
# ---------------------------------------------------------------------------

def _compute_ear(landmarks) -> float:
    """Eye Aspect Ratio from 468-point FaceMesh landmarks."""
    def _dist(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    left_v1 = _dist(landmarks[159], landmarks[145])
    left_v2 = _dist(landmarks[158], landmarks[153])
    left_h = _dist(landmarks[33], landmarks[133])
    left_ear = (left_v1 + left_v2) / (2.0 * max(left_h, 1e-6))

    right_v1 = _dist(landmarks[386], landmarks[374])
    right_v2 = _dist(landmarks[385], landmarks[380])
    right_h = _dist(landmarks[362], landmarks[263])
    right_ear = (right_v1 + right_v2) / (2.0 * max(right_h, 1e-6))

    return (left_ear + right_ear) / 2.0


def _compute_mar(landmarks) -> float:
    """Mouth Aspect Ratio."""
    def _dist(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    v1 = _dist(landmarks[13], landmarks[14])
    v2 = _dist(landmarks[82], landmarks[312])
    h = _dist(landmarks[78], landmarks[308])
    return (v1 + v2) / (2.0 * max(h, 1e-6))


def _compute_brow_height(landmarks) -> float:
    """Brow-to-eye vertical distance ratio."""
    def _dist_y(a, b):
        return abs(a.y - b.y)

    left_brow_eye = _dist_y(landmarks[70], landmarks[159])
    right_brow_eye = _dist_y(landmarks[300], landmarks[386])
    return (left_brow_eye + right_brow_eye) / 2.0


def _compute_head_pose(landmarks) -> tuple[float, float]:
    """Approximate head yaw and pitch from nose-to-cheek ratios."""
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]

    nose_to_left = abs(nose.x - left_cheek.x)
    nose_to_right = abs(nose.x - right_cheek.x)
    total_w = nose_to_left + nose_to_right
    yaw = 0.0
    if total_w > 1e-6:
        yaw = (nose_to_left - nose_to_right) / total_w * 45.0

    nose_to_forehead = abs(nose.y - forehead.y)
    nose_to_chin = abs(nose.y - chin.y)
    total_h = nose_to_forehead + nose_to_chin
    pitch = 0.0
    if total_h > 1e-6:
        pitch = (nose_to_forehead - nose_to_chin) / total_h * 30.0

    return yaw, pitch


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
