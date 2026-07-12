"""Typed data schemas using Pydantic.

These schemas define the contract between Stage 0 (data prep) and downstream
stages. Annotations are serialized as JSON on disk; these models provide
validation and IDE autocomplete.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class EmotionLabel(str, Enum):
    """Gaming-specific emotion labels (primary schema, `gaming_8`).

    Class index = declaration order (0..7). Do not reorder without rebuilding
    cached features and checkpoints. See `docs/annotation_guideline.md`.
    """
    NEUTRAL = "neutral"        # 0 — baseline / idle / explanatory / silent tryhard
    HYPE = "hype"              # 1 — clutch, ace, victory adrenaline
    AMUSED = "amused"          # 2 — laughter, funny moment
    TILTED = "tilted"          # 3 — anger, frustration, ragequit
    SAD = "sad"                # 4 — loss, regret, disappointment
    SHOCKED = "shocked"        # 5 — surprise (positive or negative)
    FEAR = "fear"              # 6 — horror, jump-scare, panic
    DISGUSTED = "disgusted"    # 7 — revulsion, cringe, contempt


#: Valid values for `config.yaml`'s `labeling.merge_mode`.
LABEL_MERGE_MODES = ("none", "merge_hype_amused", "amused_as_happy")


def resolve_labels(merge_mode: str = "none") -> tuple[list[str], dict[str, int]]:
    """Resolve training class names + raw-label→index mapping for a merge mode.

    `EmotionLabel` (gaming_8, 8 classes) stays the single source of truth for
    what's actually stored in annotation JSON / cached features — raw labels
    are never rewritten. This function only controls how those 8 raw labels
    get folded into training target indices, so switching modes never
    requires re-annotating or re-extracting features.

    Modes:
        "none" (default): gaming_8 unchanged — 8 classes, identity mapping.
        "amused_as_happy": still 8 classes, same indices as "none" — only the
            display name for index 2 changes from "amused" to "happy" (hype
            stays a separate class). Cached features/checkpoints trained
            under "none" remain valid under this mode (same n_classes, same
            index assignment) — only display/log strings differ.
        "merge_hype_amused": 7 classes — "hype" and "amused" both map to a
            single "happy" index; tilted..disgusted shift down by one index
            relative to "none". n_classes changes, so this requires
            retraining from scratch (Stage 1 perception onward) — a
            checkpoint trained under "none"/"amused_as_happy" is NOT
            compatible (classifier output layer shape differs).

    Args:
        merge_mode: One of `LABEL_MERGE_MODES`.

    Returns:
        (class_names, label_to_idx): `class_names` is the ordered list of
        training class names (length = n_classes for that mode).
        `label_to_idx` maps each raw `EmotionLabel.value` string to its
        training class index under this mode.

    Raises:
        ValueError: If `merge_mode` is not a recognized value.
    """
    if merge_mode == "none":
        class_names = [e.value for e in EmotionLabel]
        label_to_idx = {e.value: i for i, e in enumerate(EmotionLabel)}
    elif merge_mode == "amused_as_happy":
        class_names = [e.value for e in EmotionLabel]
        class_names[class_names.index("amused")] = "happy"
        label_to_idx = {e.value: i for i, e in enumerate(EmotionLabel)}
    elif merge_mode == "merge_hype_amused":
        class_names = ["neutral", "happy", "tilted", "sad", "shocked", "fear", "disgusted"]
        label_to_idx = {
            "neutral": 0,
            "hype": 1,
            "amused": 1,
            "tilted": 2,
            "sad": 3,
            "shocked": 4,
            "fear": 5,
            "disgusted": 6,
        }
    else:
        raise ValueError(
            f"Unknown labeling.merge_mode '{merge_mode}'. Available: {LABEL_MERGE_MODES}"
        )
    return class_names, label_to_idx


class EkmanLabel(str, Enum):
    """Ekman 7 emotion labels (alternative schema for ablation)."""
    VUI = "vui"
    BUON = "buon"
    TUC_GIAN = "tuc_gian"
    SO_HAI = "so_hai"
    NGAC_NHIEN = "ngac_nhien"
    GHE_TOM = "ghe_tom"
    TRUNG_TINH = "trung_tinh"


class GameGenre(str, Enum):
    """Game genre categories (for stratified split and per-genre eval)."""
    MOBA = "moba"
    FPS = "fps"
    HORROR = "horror"
    CASUAL = "casual"
    RPG = "rpg"
    MOBILE = "mobile"


class Clip(BaseModel):
    """A single video clip from a livestream/review."""

    clip_id: str
    source_url: str
    streamer: str
    genre: GameGenre
    video_path: Path
    duration_seconds: float
    resolution: tuple[int, int]
    fps: float
    created_at: datetime

    @field_validator("video_path", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v)

    @field_validator("duration_seconds")
    @classmethod
    def positive_duration(cls, v: float) -> float:
        """Ensure duration is positive."""
        if v <= 0:
            raise ValueError(f"duration_seconds must be positive, got {v}")
        return v


class WebcamBBox(BaseModel):
    """Detected webcam region in normalized coords (0–1)."""

    xmin: float
    ymin: float
    width: float
    height: float
    stability_score: float

    @model_validator(mode="after")
    def check_bounds(self) -> "WebcamBBox":
        """Validate all coordinates are in [0, 1]."""
        for field in ("xmin", "ymin", "width", "height", "stability_score"):
            val = getattr(self, field)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{field}={val} must be in [0, 1]")
        return self


class AnnotatorAgent(BaseModel):
    """Per-modality description from one agent in the multi-agent pipeline."""

    agent_name: str
    model_version: str
    output: str
    metadata: dict = {}


class Annotation(BaseModel):
    """Full multimodal annotation for a single clip."""

    clip_id: str
    emotion_label: EmotionLabel
    ekman_label: EkmanLabel | None = None
    genre: GameGenre

    face_aus: dict[str, float]
    peak_frame_idx: int
    webcam_bbox: WebcamBBox | None = None
    face_description: str = ""
    visual_objective_desc: str
    audio_tone_desc: str
    transcript: str
    reasoning: str

    annotators: list[AnnotatorAgent]
    human_verified: bool = False
    human_reviewer: str | None = None
    cohens_kappa: float | None = None
    code_switching_ratio: float = 0.0
    source_language: Literal["vi", "en"] = "vi"
    asr_detected_language: str | None = None
    text_detected_language: str | None = None
    language_detect_confidence: float | None = None
    language_mismatch: bool = False
    created_at: datetime

    def save(self, path: Path) -> None:
        """Serialize annotation to JSON file.

        Args:
            path: Output path. Parent dirs created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Annotation":
        """Load annotation from JSON file.

        Args:
            path: Path to annotation JSON.

        Returns:
            Validated Annotation instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class MultimodalFeatures(BaseModel):
    """Cached encoder outputs (used by training without re-running encoders)."""

    clip_id: str
    h_audio_path: Path
    h_face_path: Path
    h_context_path: Path
    h_text_path: Path
    audio_shape: tuple[int, ...]
    face_shape: tuple[int, ...]
    context_shape: tuple[int, ...]
    text_shape: tuple[int, ...]
    has_facecam: bool
    extracted_at: datetime

    @field_validator("h_audio_path", "h_face_path", "h_context_path", "h_text_path", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v)

    def save(self, path: Path) -> None:
        """Serialize features metadata to JSON.

        Args:
            path: Output JSON path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="python")
        for k in ("h_audio_path", "h_face_path", "h_context_path", "h_text_path"):
            data[k] = str(data[k])
        data["extracted_at"] = data["extracted_at"].isoformat()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MultimodalFeatures":
        """Load features metadata from JSON.

        Args:
            path: Path to features metadata JSON.

        Returns:
            Validated MultimodalFeatures instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Features metadata not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
