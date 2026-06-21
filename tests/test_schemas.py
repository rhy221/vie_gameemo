"""Tests for data schemas (Pydantic models)."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from vie_gameemo.data.schemas import (
    Annotation,
    AnnotatorAgent,
    Clip,
    EmotionLabel,
    EkmanLabel,
    GameGenre,
    MultimodalFeatures,
    WebcamBBox,
)


class TestEmotionLabel:
    def test_all_8_labels(self):
        assert len(EmotionLabel) == 8

    def test_label_values(self):
        expected = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]
        assert [e.value for e in EmotionLabel] == expected

    def test_label_from_value(self):
        assert EmotionLabel("neutral") == EmotionLabel.NEUTRAL
        assert EmotionLabel("hype") == EmotionLabel.HYPE


class TestGameGenre:
    def test_all_genres(self):
        assert len(GameGenre) == 6
        assert GameGenre("moba") == GameGenre.MOBA


class TestWebcamBBox:
    def test_valid_bbox(self):
        bbox = WebcamBBox(xmin=0.7, ymin=0.7, width=0.2, height=0.2, stability_score=0.8)
        assert bbox.xmin == 0.7

    def test_invalid_bbox_raises(self):
        with pytest.raises(Exception):
            WebcamBBox(xmin=1.5, ymin=0.0, width=0.2, height=0.2, stability_score=0.5)

    def test_negative_value_raises(self):
        with pytest.raises(Exception):
            WebcamBBox(xmin=-0.1, ymin=0.0, width=0.2, height=0.2, stability_score=0.5)


class TestClip:
    def test_valid_clip(self, tmp_path):
        clip = Clip(
            clip_id="test_001",
            source_url="https://youtube.com/watch?v=xxx",
            streamer="streamer_a",
            genre=GameGenre.MOBA,
            video_path=tmp_path / "test.mp4",
            duration_seconds=5.0,
            resolution=(1280, 720),
            fps=30.0,
            created_at=datetime.now(),
        )
        assert clip.clip_id == "test_001"

    def test_negative_duration_raises(self, tmp_path):
        with pytest.raises(Exception):
            Clip(
                clip_id="test", source_url="url", streamer="s",
                genre=GameGenre.FPS, video_path=tmp_path / "x.mp4",
                duration_seconds=-1.0, resolution=(720, 480),
                fps=30.0, created_at=datetime.now(),
            )

    def test_string_path_coercion(self, tmp_path):
        clip = Clip(
            clip_id="test", source_url="url", streamer="s",
            genre=GameGenre.FPS, video_path=str(tmp_path / "x.mp4"),
            duration_seconds=5.0, resolution=(720, 480),
            fps=30.0, created_at=datetime.now(),
        )
        assert isinstance(clip.video_path, Path)


class TestAnnotation:
    @pytest.fixture
    def sample_annotation(self):
        return Annotation(
            clip_id="clip_001",
            emotion_label=EmotionLabel.HYPE,
            genre=GameGenre.MOBA,
            face_aus={"AU6": 3.5, "AU12": 4.2},
            peak_frame_idx=10,
            visual_objective_desc="Streamer celebrating after winning.",
            audio_tone_desc="High pitch, fast speaking, laughing.",
            transcript="GG! We won!",
            reasoning="The streamer is excited after winning the match.",
            annotators=[
                AnnotatorAgent(agent_name="qwen_vl", model_version="7B", output="scene desc"),
            ],
            created_at=datetime.now(),
        )

    def test_valid_annotation(self, sample_annotation):
        assert sample_annotation.emotion_label == EmotionLabel.HYPE
        assert sample_annotation.genre == GameGenre.MOBA

    def test_save_and_load(self, sample_annotation, tmp_path):
        path = tmp_path / "ann.json"
        sample_annotation.save(path)
        assert path.exists()

        loaded = Annotation.load(path)
        assert loaded.clip_id == sample_annotation.clip_id
        assert loaded.emotion_label == sample_annotation.emotion_label

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Annotation.load(tmp_path / "nonexistent.json")

    def test_optional_ekman_label(self, sample_annotation):
        assert sample_annotation.ekman_label is None

    def test_with_webcam_bbox(self):
        ann = Annotation(
            clip_id="clip_002",
            emotion_label=EmotionLabel.NEUTRAL,
            genre=GameGenre.FPS,
            face_aus={},
            peak_frame_idx=5,
            webcam_bbox=WebcamBBox(xmin=0.7, ymin=0.7, width=0.2, height=0.2, stability_score=0.9),
            visual_objective_desc="",
            audio_tone_desc="",
            transcript="",
            reasoning="",
            annotators=[],
            created_at=datetime.now(),
        )
        assert ann.webcam_bbox is not None
        assert ann.webcam_bbox.stability_score == 0.9


class TestMultimodalFeatures:
    def test_save_and_load(self, tmp_path):
        feat = MultimodalFeatures(
            clip_id="clip_001",
            h_audio_path=tmp_path / "audio.pt",
            h_face_path=tmp_path / "face.pt",
            h_context_path=tmp_path / "ctx.pt",
            h_text_path=tmp_path / "text.pt",
            audio_shape=(1, 64, 768),
            face_shape=(1, 16, 768),
            context_shape=(1, 16, 768),
            text_shape=(1, 1, 768),
            has_facecam=True,
            extracted_at=datetime.now(),
        )
        path = tmp_path / "feat.json"
        feat.save(path)
        assert path.exists()

        loaded = MultimodalFeatures.load(path)
        assert loaded.clip_id == "clip_001"
        assert loaded.has_facecam is True
