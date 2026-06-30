"""Smoke tests for LLM-1 Faithful Explainer components.

Tests that:
  1. EmotionClassifier.forward(return_penultimate=True) returns correct shapes
  2. ModalAdapter with penult produces correct soft token shape
  3. CueExtractor produces valid output from mock data
  4. MLP + encoders have no gradient (frozen assertion)
"""

import torch
import pytest


class TestClassifierPenultimate:
    """B1: EmotionClassifier return_penultimate."""

    def test_default_returns_logits_only(self):
        from vie_gameemo.classifiers.mlp import EmotionClassifier
        clf = EmotionClassifier(d_model=768, hidden_dim=256, n_classes=8)
        x = torch.randn(2, 10, 768)
        out = clf(x)
        assert out.shape == (2, 8)

    def test_return_penultimate_shapes(self):
        from vie_gameemo.classifiers.mlp import EmotionClassifier
        clf = EmotionClassifier(d_model=768, hidden_dim=256, n_classes=8)
        x = torch.randn(2, 10, 768)
        logits, penult = clf(x, return_penultimate=True)
        assert logits.shape == (2, 8)
        assert penult.shape == (2, 256)

    def test_penult_is_pre_head(self):
        from vie_gameemo.classifiers.mlp import EmotionClassifier
        clf = EmotionClassifier(d_model=768, hidden_dim=256, n_classes=8)
        clf.eval()
        x = torch.randn(1, 5, 768)
        logits, penult = clf(x, return_penultimate=True)
        expected_logits = clf.net[3](penult)
        assert torch.allclose(logits, expected_logits, atol=1e-6)

    def test_2d_input(self):
        from vie_gameemo.classifiers.mlp import EmotionClassifier
        clf = EmotionClassifier(d_model=768, hidden_dim=256, n_classes=8)
        x = torch.randn(4, 768)
        logits, penult = clf(x, return_penultimate=True)
        assert logits.shape == (4, 8)
        assert penult.shape == (4, 256)

    def test_backward_compat_no_flag(self):
        from vie_gameemo.classifiers.mlp import EmotionClassifier
        clf = EmotionClassifier(d_model=768, hidden_dim=256, n_classes=8)
        x = torch.randn(2, 10, 768)
        out = clf(x, return_penultimate=False)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8)


class TestModalAdapterPenult:
    """B2: ModalAdapter with proj_penult."""

    def test_without_penult_backward_compat(self):
        from vie_gameemo.llm.modal_adapter import ModalAdapter
        adapter = ModalAdapter(d_fusion=768, d_llm=128, d_penult=256)
        fusion_emb = torch.randn(2, 10, 768)
        tokens, mask = adapter(fusion_emb)
        assert tokens.shape == (2, 10, 128)
        assert mask.shape == (2, 10)

    def test_with_penult_prepends(self):
        from vie_gameemo.llm.modal_adapter import ModalAdapter
        adapter = ModalAdapter(d_fusion=768, d_llm=128, d_penult=256)
        fusion_emb = torch.randn(2, 10, 768)
        penult = torch.randn(2, 256)
        tokens, mask = adapter(fusion_emb, penult=penult)
        assert tokens.shape == (2, 11, 128)  # 1 penult + 10 fusion
        assert mask.shape == (2, 11)
        assert mask[:, 0].all()  # penult always attended

    def test_with_penult_and_modalities(self):
        from vie_gameemo.llm.modal_adapter import ModalAdapter
        adapter = ModalAdapter(d_fusion=768, d_llm=128, d_penult=256)
        fusion_emb = torch.randn(2, 10, 768)
        penult = torch.randn(2, 256)
        audio = torch.randn(2, 64, 768)
        face = torch.randn(2, 1, 768)
        tokens, mask = adapter(fusion_emb, penult=penult, audio=audio, face=face)
        expected_t = 1 + 10 + 64 + 1  # penult + fusion + audio + face
        assert tokens.shape == (2, expected_t, 128)

    def test_penult_3d_input(self):
        from vie_gameemo.llm.modal_adapter import ModalAdapter
        adapter = ModalAdapter(d_fusion=768, d_llm=128, d_penult=256)
        fusion_emb = torch.randn(2, 10, 768)
        penult = torch.randn(2, 1, 256)  # already 3D
        tokens, mask = adapter(fusion_emb, penult=penult)
        assert tokens.shape == (2, 11, 128)


class TestCueExtractor:
    """B3: CueExtractor produces valid output."""

    def test_text_cue_with_content(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(cache_dir="/tmp/test_cue_cache")
        cue, attrs = ext._text_cue("ACE rồi bro! kill headshot quá đỉnh!")
        assert "exclamation" in cue
        assert "game_term" in cue
        assert len(attrs) == 4
        assert all(0 <= a <= 1.0 for a in attrs)

    def test_text_cue_empty(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(cache_dir="/tmp/test_cue_cache")
        cue, attrs = ext._text_cue("")
        assert cue == "no_speech"
        assert attrs == [0.0, 0.0, 0.0, 0.0]

    def test_n_attrs(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor()
        assert ext.n_attrs == 15

    def test_voice_cue_fallback(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(cache_dir="/tmp/nonexistent_cache")
        cue, attrs = ext._voice_cue("nonexistent_clip", "hello world")
        assert cue == "no_audio"
        assert len(attrs) == 3

    def test_face_cue_no_face(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(cache_dir="/tmp/nonexistent_cache")
        cue, attrs = ext._face_cue("nonexistent_clip", has_face=False)
        assert cue == "no_webcam"
        assert len(attrs) == 5

    def test_scene_cue_fallback(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(cache_dir="/tmp/nonexistent_cache")
        cue, attrs = ext._scene_cue("nonexistent_clip")
        assert cue == "no_visual"
        assert len(attrs) == 3


class TestFrozenAssertion:
    """Verify MLP + fusion don't get gradient in LLM-1 training setup."""

    def test_classifier_frozen(self):
        from vie_gameemo.classifiers.mlp import EmotionClassifier
        clf = EmotionClassifier(d_model=768, hidden_dim=256, n_classes=8)
        for p in clf.parameters():
            p.requires_grad = False

        x = torch.randn(2, 10, 768)
        logits, penult = clf(x, return_penultimate=True)
        assert not any(p.requires_grad for p in clf.parameters())


class TestGHead:
    """g_head reconstruction MLP."""

    def test_forward_shape(self):
        from vie_gameemo.training.llm1_explanation import GHead
        g = GHead(d_input=768, hidden_dim=128, n_attrs=15)
        x = torch.randn(4, 768)
        out = g(x)
        assert out.shape == (4, 15)


class TestParseOutput:
    """LLM1Explainer.parse_output handles both formats."""

    def test_cue_format(self):
        from vie_gameemo.llm.llm1_explainer import LLM1Explainer
        raw = "Cues: face: eyes=wide; voice: pitch=high. Emotion: hype."
        reasoning, answer, valid = LLM1Explainer.parse_output(raw, "neutral")
        assert valid
        assert answer == "hype"
        assert "eyes=wide" in reasoning

    def test_think_answer_format(self):
        from vie_gameemo.llm.llm1_explainer import LLM1Explainer
        raw = "<think>AU12 cao</think><answer>amused</answer>"
        reasoning, answer, valid = LLM1Explainer.parse_output(raw, "neutral")
        assert valid
        assert answer == "amused"
        assert "AU12" in reasoning

    def test_fallback(self):
        from vie_gameemo.llm.llm1_explainer import LLM1Explainer
        raw = "some random text"
        reasoning, answer, valid = LLM1Explainer.parse_output(raw, "neutral")
        assert not valid
        assert answer == "neutral"
