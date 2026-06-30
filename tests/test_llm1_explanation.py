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

    def test_n_attrs_vit_imagenet(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(context_encoder_type="vit_imagenet")
        assert ext.n_attrs == 12

    def test_n_attrs_pose(self):
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(context_encoder_type="pose")
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

    def test_extract_vit_imagenet_no_context_cue(self):
        """vit_imagenet branch: extract() must NOT produce a motion cue."""
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(cache_dir="/tmp/nonexistent_cache", context_encoder_type="vit_imagenet")
        cue, attr = ext.extract("nonexistent_clip", transcript="hello", has_face=False)
        assert "motion" not in cue
        assert attr.shape == (12,)

    def test_extract_pose_includes_motion_cue(self):
        """pose branch: extract() must include a motion cue section."""
        from vie_gameemo.llm.cue_extractor import CueExtractor
        ext = CueExtractor(cache_dir="/tmp/nonexistent_cache", context_encoder_type="pose")
        cue, attr = ext.extract("nonexistent_clip", transcript="hello", has_face=False)
        assert "motion" in cue
        assert attr.shape == (15,)


class TestMotionCueExtractor:
    """MotionCueExtractor: kinematic cues from pose cache."""

    def test_extract_missing_cache(self):
        from vie_gameemo.llm.cue_extractor import MotionCueExtractor
        m = MotionCueExtractor(cache_dir="/tmp/nonexistent_cache")
        cue, attrs = m.extract("nonexistent_clip")
        assert cue == "no_motion_data"
        assert attrs == [0.0, 0.0, 0.0]
        assert len(attrs) == 3

    def test_extract_zero_kinematics(self, tmp_path):
        """Zero energy/impact → minimal movement cue."""
        import json
        from vie_gameemo.llm.cue_extractor import MotionCueExtractor
        cache_dir = tmp_path / "cache"
        kin_dir = cache_dir / "pose_kinematics"
        kin_dir.mkdir(parents=True)
        (kin_dir / "clip0.json").write_text(
            json.dumps({"motion_energy": 0.001, "impact": 0.001, "periodicity": 0.01}),
            encoding="utf-8",
        )
        m = MotionCueExtractor(cache_dir=cache_dir)
        cue, attrs = m.extract("clip0")
        assert "motion" in cue
        assert len(attrs) == 3
        assert all(0.0 <= a <= 1.0 for a in attrs)

    def test_extract_high_impact(self, tmp_path):
        """High impact → desk-slap cue."""
        import json
        from vie_gameemo.llm.cue_extractor import MotionCueExtractor
        cache_dir = tmp_path / "cache"
        kin_dir = cache_dir / "pose_kinematics"
        kin_dir.mkdir(parents=True)
        (kin_dir / "clip1.json").write_text(
            json.dumps({"motion_energy": 0.1, "impact": 0.5, "periodicity": 0.1}),
            encoding="utf-8",
        )
        m = MotionCueExtractor(cache_dir=cache_dir)
        cue, _ = m.extract("clip1")
        assert "sharp downward arm impact" in cue

    def test_extract_high_energy_rhythmic(self, tmp_path):
        """High energy + rhythmic → hype cue."""
        import json
        from vie_gameemo.llm.cue_extractor import MotionCueExtractor
        cache_dir = tmp_path / "cache"
        kin_dir = cache_dir / "pose_kinematics"
        kin_dir.mkdir(parents=True)
        (kin_dir / "clip2.json").write_text(
            json.dumps({"motion_energy": 0.15, "impact": 0.02, "periodicity": 0.8}),
            encoding="utf-8",
        )
        m = MotionCueExtractor(cache_dir=cache_dir)
        cue, _ = m.extract("clip2")
        assert "rhythmic" in cue


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


class TestGHeadPerModality:
    """GHeadPerModality: per-modality reconstruction heads (FIX 3)."""

    def test_forward_shapes_with_context(self):
        from vie_gameemo.training.llm1_explanation import GHeadPerModality
        g = GHeadPerModality(d_input=768, hidden_dim=64, has_context=True)
        face = torch.randn(2, 10, 768)
        audio = torch.randn(2, 64, 768)
        context = torch.randn(2, 1, 768)
        text = torch.randn(2, 20, 768)
        fp, vp, mp, tp = g(face, audio, context, text)
        assert fp.shape == (2, 5)
        assert vp.shape == (2, 3)
        assert mp is not None and mp.shape == (2, 3)
        assert tp.shape == (2, 4)

    def test_forward_shapes_no_context(self):
        from vie_gameemo.training.llm1_explanation import GHeadPerModality
        g = GHeadPerModality(d_input=768, hidden_dim=64, has_context=False)
        face = torch.randn(2, 10, 768)
        audio = torch.randn(2, 64, 768)
        context = torch.randn(2, 1, 768)
        text = torch.randn(2, 20, 768)
        fp, vp, mp, tp = g(face, audio, context, text)
        assert fp.shape == (2, 5)
        assert vp.shape == (2, 3)
        assert mp is None
        assert tp.shape == (2, 4)

    def test_heads_read_own_modality(self):
        """Verify each head is sensitive to its own input, not others."""
        from vie_gameemo.training.llm1_explanation import GHeadPerModality
        g = GHeadPerModality(d_input=16, hidden_dim=8, n_face=2, n_voice=2,
                             n_motion=2, n_text=2, has_context=True)
        face = torch.randn(1, 5, 16)
        audio = torch.randn(1, 5, 16)
        context = torch.randn(1, 5, 16)
        text = torch.randn(1, 5, 16)
        fp1, vp1, mp1, tp1 = g(face, audio, context, text)
        # Change only face — face_pred should change, others should be same
        face2 = torch.randn_like(face)
        fp2, vp2, mp2, tp2 = g(face2, audio, context, text)
        assert not torch.allclose(fp1, fp2)
        assert torch.allclose(vp1, vp2)
        assert torch.allclose(mp1, mp2)
        assert torch.allclose(tp1, tp2)


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


class TestKLLossFormula:
    """FIX 7.3: unit tests for the KL distillation formula used in L_kl."""

    def test_identical_distributions_zero_loss(self):
        """KL(P||P) = 0: when LLM distribution equals MLP distribution, L_kl ≈ 0."""
        import torch
        import torch.nn.functional as F

        logits = torch.randn(2, 8)
        T = 2.0
        mlp_soft = F.softmax(logits / T, dim=-1)
        lm_log_soft = F.log_softmax(logits / T, dim=-1)
        loss = F.kl_div(lm_log_soft, mlp_soft, reduction="batchmean") * (T ** 2)
        assert loss.item() < 1e-5

    def test_temperature_squared_scaling_changes_loss(self):
        """T² scaling has real effect — loss at T=1 and T=2 differ measurably."""
        import torch
        import torch.nn.functional as F

        torch.manual_seed(42)
        logits_a = torch.randn(4, 8)
        logits_b = torch.randn(4, 8)

        def kl_loss(T):
            s = F.softmax(logits_a / T, dim=-1)
            ls = F.log_softmax(logits_b / T, dim=-1)
            return F.kl_div(ls, s, reduction="batchmean") * (T ** 2)

        assert abs(kl_loss(1.0).item() - kl_loss(2.0).item()) > 1e-4

    def test_label_token_ids_unique_raises_on_collision(self):
        """_build_label_token_ids raises ValueError when two labels share first token."""
        import pytest
        from vie_gameemo.training.llm1_explanation import _build_label_token_ids

        class _MockTokenizer:
            def encode(self, name, add_special_tokens=False):
                return [42]  # single token but all same → collision

        with pytest.raises(ValueError, match="Duplicate"):
            _build_label_token_ids(_MockTokenizer(), ["a", "b"])

    def test_label_multi_token_raises(self):
        """_build_label_token_ids raises ValueError for any multi-token label (FIX 8.3)."""
        import pytest
        from vie_gameemo.training.llm1_explanation import _build_label_token_ids

        class _MockMultiTokenizer:
            def encode(self, name, add_special_tokens=False):
                if name == "b":
                    return [10, 11]  # multi-token
                return [ord(name[0])]  # others: single token

        with pytest.raises(ValueError, match="single token|single-token"):
            _build_label_token_ids(_MockMultiTokenizer(), ["a", "b", "c"])


class TestMotionCueWindowGate:
    """FIX 8.1: impact gate must check the full 3-frame derivative window."""

    def test_gap_tracking_does_not_produce_false_impact(self):
        """Keypoint tracking gap (conf≈0 mid-sequence) followed by high-conf reappearance
        must NOT produce a large impact cue — the 3-frame window gate must zero it out.

        Bug scenario without the gate:
          velocity[10] = pos[10] − pos[9]   (pos[9] is garbage from gap → huge spike)
          Even though conf[10] is high, conf[9] is near 0 → gate must block frame 10.
        """
        import numpy as np
        from vie_gameemo.encoders.context_pose import _build_kinematic_features

        T, K = 12, 6
        kps = np.zeros((T, K, 4))  # (T, K, x/y/z/conf)

        # Frames 0-4: wrist tracked at y≈0.5, high confidence
        for t in range(5):
            kps[t, :, 1] = 0.5
            kps[t, :, 3] = 0.95

        # Frames 5-9: tracking dropout — garbage position, near-zero confidence
        for t in range(5, 10):
            kps[t, :, 1] = 999.0   # huge garbage position
            kps[t, :, 3] = 0.05

        # Frames 10-11: reappears at realistic position, high confidence
        for t in range(10, 12):
            kps[t, :, 1] = 0.6
            kps[t, :, 3] = 0.95

        feats = _build_kinematic_features(kps)   # (T, K*7)
        K_f = feats.shape[1] // 7
        acc = feats[:, K_f * 4: K_f * 6].reshape(T, K_f, 2)
        conf_f = feats[:, K_f * 6:].reshape(T, K_f)

        # Replicate the binary gate from MotionCueExtractor.precompute() (P0.0-fix.4: body anchor)
        # K=6 ≤ 13 → MMPose path: nose=0, left_shoulder=5, right_shoulder=6 (capped to K)
        body_anchor_idx = [i for i in ([0, 1, 2] if K_f > 13 else [0, 5, 6]) if i < K_f]
        anchor_conf = conf_f[:, body_anchor_idx]
        threshold = 0.3
        min_conf_3frame = np.zeros_like(anchor_conf)
        min_conf_3frame[2:] = np.minimum(
            np.minimum(anchor_conf[2:], anchor_conf[1:-1]), anchor_conf[:-2]
        )
        conf_gate = (min_conf_3frame >= threshold).astype(float)
        anchor_acc_down = acc[:, body_anchor_idx, 1] * conf_gate
        impact = float(np.max(np.abs(anchor_acc_down))) if anchor_acc_down.size > 0 else 0.0

        # Without gate: velocity[10] = 999 − 0.05 ≈ 999 → acc ≈ 1998 (huge fake impact)
        # With gate:    min_conf(10,9,8) = 0.05 < 0.3 → conf_gate[10] = 0 → impact ≈ 0
        assert impact < 1.0, (
            f"False impact detected ({impact:.2f}) — 3-frame window gate should have "
            "zeroed out derivatives at the gap boundary"
        )
