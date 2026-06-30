"""CueExtractor: label-orthogonal multimodal cue extraction.

Extracts fine-grained, non-emotional descriptors from 4 modalities:
  - Face: MediaPipe FaceMesh geometric ratios (EAR, MAR, brow height, head pose)
  - Voice: librosa prosody (pitch, energy, speaking rate)
  - Context: OpenCV visual stats (brightness, color variance, edge density)
  - Text: regex-based lexicon hits (exclamations, negative words, game terms)

Cues are strictly orthogonal to emotion labels — they describe physical
properties, not emotions. This forces the LLM to read raw modality tokens
(tap A) to generate the cue portion of its output.

Usage:
    extractor = CueExtractor(cache_dir="data/cache")
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

_BRIGHTNESS_BINS = [(85, "dark"), (170, "moderate"), (float("inf"), "bright")]
_COLOR_VAR_BINS = [(500, "muted"), (1500, "moderate"), (float("inf"), "colorful")]
_EDGE_BINS = [(0.05, "sparse"), (0.12, "moderate"), (float("inf"), "busy")]

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


class CueExtractor:
    """Extracts label-orthogonal cues from 4 modalities.

    Precompute once, then read from cache during training.
    """

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.face_geo_dir = self.cache_dir / "face_geo"
        self.prosody_dir = self.cache_dir / "prosody"
        self.visual_stats_dir = self.cache_dir / "visual_stats"

    # ------------------------------------------------------------------
    # Precompute (run once before training)
    # ------------------------------------------------------------------

    def precompute_all(
        self,
        faces_dir: str | Path,
        audios_dir: str | Path,
        frames_dir: str | Path,
        clip_ids: list[str] | None = None,
    ) -> None:
        """Batch-extract face geometry, prosody, and visual stats for all clips."""
        faces_dir = Path(faces_dir)
        audios_dir = Path(audios_dir)
        frames_dir = Path(frames_dir)

        for d in (self.face_geo_dir, self.prosody_dir, self.visual_stats_dir):
            d.mkdir(parents=True, exist_ok=True)

        if clip_ids is None:
            clip_ids = sorted({p.stem for p in audios_dir.glob("*.wav")})

        logger.info("Precomputing cues for %d clips", len(clip_ids))

        for cid in clip_ids:
            self._precompute_face_geo(cid, faces_dir)
            self._precompute_prosody(cid, audios_dir)
            self._precompute_visual_stats(cid, frames_dir)

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
            rate = 0.0  # placeholder, filled in during extract() from transcript

            _save_json(out, {"f0": mean_f0, "rms": mean_rms, "rate": rate, "duration": duration})
        except Exception as e:
            logger.debug("Prosody failed for %s: %s", clip_id, e)
            _save_json(out, {"f0": 0, "rms": 0, "rate": 0, "duration": 5.0})

    def _precompute_visual_stats(self, clip_id: str, frames_dir: Path) -> None:
        out = self.visual_stats_dir / f"{clip_id}.json"
        if out.exists():
            return

        frame_dir = frames_dir / clip_id
        if not frame_dir.exists():
            _save_json(out, {"brightness": 128, "color_var": 500, "edge_density": 0.05})
            return

        try:
            import cv2

            frames = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
            if not frames:
                _save_json(out, {"brightness": 128, "color_var": 500, "edge_density": 0.05})
                return

            mid = frames[len(frames) // 2]
            img = cv2.imread(str(mid))
            if img is None:
                _save_json(out, {"brightness": 128, "color_var": 500, "edge_density": 0.05})
                return

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_var = float(np.var(hsv[:, :, 0].astype(float)))

            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(np.mean(edges > 0))

            _save_json(out, {
                "brightness": brightness,
                "color_var": color_var,
                "edge_density": edge_density,
            })
        except Exception as e:
            logger.debug("Visual stats failed for %s: %s", clip_id, e)
            _save_json(out, {"brightness": 128, "color_var": 500, "edge_density": 0.05})

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
            (cue_text, attr_vector) where attr_vector is a 15-d float tensor.
        """
        face_cue, face_attrs = self._face_cue(clip_id, has_face)
        voice_cue, voice_attrs = self._voice_cue(clip_id, transcript)
        scene_cue, scene_attrs = self._scene_cue(clip_id)
        text_cue, text_attrs = self._text_cue(transcript)

        cue_text = f"face: {face_cue}; voice: {voice_cue}; scene: {scene_cue}; text: {text_cue}"

        attr_vector = torch.tensor(
            face_attrs + voice_attrs + scene_attrs + text_attrs,
            dtype=torch.float32,
        )

        return cue_text, attr_vector

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

    def _scene_cue(self, clip_id: str) -> tuple[str, list[float]]:
        data = _load_json(self.visual_stats_dir / f"{clip_id}.json")
        if data is None:
            return "no_visual", [0.0, 0.0, 0.0]

        brightness = data.get("brightness", 128)
        color_var = data.get("color_var", 500)
        edge_density = data.get("edge_density", 0.05)

        b_label = _bin_value(brightness, _BRIGHTNESS_BINS)
        c_label = _bin_value(color_var, _COLOR_VAR_BINS)
        e_label = _bin_value(edge_density, _EDGE_BINS)

        cue = f"brightness={b_label}, colors={c_label}, screen={e_label}"
        return cue, [brightness / 255.0, color_var / 2000.0, edge_density / 0.2]

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

    @property
    def n_attrs(self) -> int:
        return 15


# ------------------------------------------------------------------
# MediaPipe landmark helpers
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------

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
