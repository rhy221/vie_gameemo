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

    def test_default_source_language_vi(self, sample_annotation):
        """Old VI clips without source_language should default to 'vi'."""
        assert sample_annotation.source_language == "vi"

    def test_source_language_en(self):
        ann = Annotation(
            clip_id="clip_en_001",
            emotion_label=EmotionLabel.FEAR,
            genre=GameGenre.HORROR,
            face_aus={},
            peak_frame_idx=3,
            visual_objective_desc="Player in dark corridor.",
            audio_tone_desc="Screaming, high pitch.",
            transcript="Oh my god! What was that?!",
            reasoning="Jump scare reaction.",
            annotators=[],
            source_language="en",
            created_at=datetime.now(),
        )
        assert ann.source_language == "en"
        assert ann.language_mismatch is False

    def test_language_detect_fields(self):
        ann = Annotation(
            clip_id="clip_detect",
            emotion_label=EmotionLabel.NEUTRAL,
            genre=GameGenre.CASUAL,
            face_aus={},
            peak_frame_idx=0,
            visual_objective_desc="",
            audio_tone_desc="",
            transcript="test",
            reasoning="",
            annotators=[],
            source_language="vi",
            asr_detected_language="en",
            text_detected_language="en",
            language_detect_confidence=0.85,
            language_mismatch=True,
            created_at=datetime.now(),
        )
        assert ann.asr_detected_language == "en"
        assert ann.text_detected_language == "en"
        assert ann.language_detect_confidence == 0.85
        assert ann.language_mismatch is True

    def test_backward_compat_old_vi_clip_no_lang_fields(self, tmp_path):
        """Simulate loading a JSON from before bilingual support was added."""
        old_json = {
            "clip_id": "old_clip",
            "emotion_label": "neutral",
            "genre": "moba",
            "face_aus": {},
            "peak_frame_idx": 0,
            "visual_objective_desc": "",
            "audio_tone_desc": "",
            "transcript": "xin chào",
            "reasoning": "",
            "annotators": [],
            "code_switching_ratio": 0.0,
            "created_at": datetime.now().isoformat(),
        }
        path = tmp_path / "old_clip.json"
        path.write_text(json.dumps(old_json), encoding="utf-8")

        loaded = Annotation.load(path)
        assert loaded.source_language == "vi"
        assert loaded.asr_detected_language is None
        assert loaded.language_mismatch is False

    def test_save_load_roundtrip_with_language(self, tmp_path):
        ann = Annotation(
            clip_id="clip_rt",
            emotion_label=EmotionLabel.AMUSED,
            genre=GameGenre.FPS,
            face_aus={"AU12": 2.0},
            peak_frame_idx=5,
            visual_objective_desc="desc",
            audio_tone_desc="tone",
            transcript="let's go!",
            reasoning="reason",
            annotators=[],
            source_language="en",
            asr_detected_language="en",
            text_detected_language="en",
            language_detect_confidence=0.92,
            language_mismatch=False,
            created_at=datetime.now(),
        )
        path = tmp_path / "rt.json"
        ann.save(path)
        loaded = Annotation.load(path)
        assert loaded.source_language == "en"
        assert loaded.asr_detected_language == "en"
        assert loaded.language_detect_confidence == 0.92


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
