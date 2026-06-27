"""Offline batch inference.

Loads a trained checkpoint, runs the full pipeline (encoders → fusion →
classifier → optional LLM), and writes results to JSON.
"""

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

_EMOTION_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


def batch_inference(
    clip_paths: list[Path],
    checkpoint: Path,
    cfg: SimpleNamespace,
    output_json: Path,
    include_llm_explanation: bool = True,
    use_cached_features: bool = True,
) -> Path:
    """Run batch inference on multiple clips.

    For each clip:
        1. Load cached features if available (and use_cached_features=True),
           otherwise run encoders inline.
        2. Forward through fusion + classifier → predicted label + confidence.
        3. Optionally call the active LLM setup for reasoning explanation.
        4. Write results JSON.

    Args:
        clip_paths: List of video clip paths (mp4) or feature dirs.
        checkpoint: Trained perception checkpoint (.pt file).
        cfg: Full config namespace.
        output_json: Where to write results JSON.
        include_llm_explanation: If True, invoke LLM for reasoning text.
        use_cached_features: If True, look for precomputed .pt features first.

    Returns:
        Path to output JSON.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion, classifier = _load_model(checkpoint, cfg, device)

    llm_reasoner = None
    if include_llm_explanation:
        llm_reasoner = _load_llm(cfg)

    results = []
    for clip_path in clip_paths:
        clip_path = Path(clip_path)
        clip_id = clip_path.stem
        t0 = time.perf_counter()
        logger.info("Inferring: %s", clip_path)

        try:
            features = _get_features(clip_path, cfg, use_cached_features)
            prediction = _forward(fusion, classifier, features, device)

            reasoning = ""
            if llm_reasoner is not None:
                evidence = _features_to_evidence(features, prediction, clip_path, cfg)
                llm_out = llm_reasoner.reason(evidence)
                reasoning = llm_out.reasoning

            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append({
                "clip_id": clip_id,
                "clip_path": str(clip_path),
                "predicted_label": prediction["label"],
                "confidence": prediction["confidence"],
                "class_scores": prediction["class_scores"],
                "reasoning": reasoning,
                "latency_ms": round(elapsed_ms, 1),
            })
        except Exception as exc:
            logger.warning("Failed to process %s: %s", clip_path, exc)
            results.append({
                "clip_id": clip_id,
                "clip_path": str(clip_path),
                "error": str(exc),
            })

    if llm_reasoner is not None and hasattr(llm_reasoner, "unload"):
        llm_reasoner.unload()

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Batch inference done. %d clips → %s", len(results), output_json)
    return output_json


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint: Path, cfg: SimpleNamespace, device: torch.device):
    """Load fusion + classifier from perception checkpoint."""
    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.fusion import get_fusion
    from vie_gameemo.training.perception import load_checkpoint

    fcfg = cfg.fusion
    ccfg = cfg.classifier

    fusion = get_fusion(
        fcfg.type,
        d_model=fcfg.d_model,
        n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=True,
    ).to(device)
    classifier = EmotionClassifier(
        d_model=fcfg.d_model,
        hidden_dim=ccfg.hidden_dim,
        n_classes=ccfg.n_classes,
        dropout=ccfg.dropout,
    ).to(device)

    load_checkpoint(checkpoint, fusion, classifier)
    fusion.eval()
    classifier.eval()
    return fusion, classifier


def _load_llm(cfg: SimpleNamespace):
    """Load the active LLM setup specified in config."""
    llm_cfg = cfg.llm
    active = getattr(llm_cfg, "active_setup", "llm1")

    cognition_ckpt = getattr(llm_cfg, "cognition_checkpoint", None)

    if active == "llm1":
        from vie_gameemo.llm.llm1_explainer import LLM1Explainer
        return LLM1Explainer(
            model_name=llm_cfg.base_model.name,
            quantization=llm_cfg.base_model.quantization,
            modal_adapter_ckpt=cognition_ckpt,
        )
    elif active == "llm2":
        from vie_gameemo.llm.llm2_coreasoner import LLM2CoReasoner
        return LLM2CoReasoner(
            model_name=llm_cfg.base_model.name,
            quantization=llm_cfg.base_model.quantization,
            modal_adapter_ckpt=Path(cognition_ckpt) if cognition_ckpt else None,
        )
    elif active == "llm3":
        from vie_gameemo.llm.llm3_vlm import LLM3PureReasoner
        return LLM3PureReasoner(
            model_name=llm_cfg.base_model.name,
            quantization=llm_cfg.base_model.quantization,
            modal_adapter_ckpt=Path(cognition_ckpt) if cognition_ckpt else None,
        )
    elif active == "llm4":
        from vie_gameemo.llm.llm4_rlvr import LLM4RLVR
        return LLM4RLVR(
            base_model=llm_cfg.base_model.name,
            quantization=llm_cfg.base_model.quantization,
            modal_adapter_ckpt=Path(cognition_ckpt) if cognition_ckpt else None,
        )
    else:
        logger.warning("Unknown LLM setup '%s'; skipping LLM", active)
        return None


def _get_features(clip_path: Path, cfg: SimpleNamespace, use_cached: bool) -> dict:
    """Load cached features or extract on the fly."""
    if use_cached:
        from vie_gameemo.data.feature_cache import load_cached_features
        cached = load_cached_features(clip_path.stem, Path(cfg.paths.features))
        if cached is not None:
            return cached

    return _extract_features_inline(clip_path, cfg)


def _extract_features_inline(clip_path: Path, cfg: SimpleNamespace) -> dict:
    """Extract features from raw video using frozen encoders."""
    from vie_gameemo.preprocess.demux import extract_audio, extract_frames

    tmp_dir = Path(cfg.paths.features) / "_tmp" / clip_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    audio_path = tmp_dir / "audio.wav"
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    extract_audio(clip_path, audio_path)
    extract_frames(clip_path, frames_dir, target_fps=4)
    frame_paths = sorted(frames_dir.glob("*.jpg"))

    # Detect webcam region for context encoder
    from vie_gameemo.preprocess.webcam_detector import WebcamDetector
    detector = WebcamDetector()
    webcam_bbox = detector.detect_webcam_region(clip_path)

    features: dict = {}

    from vie_gameemo.encoders.audio_ast import ASTAudioEncoder
    audio_enc = ASTAudioEncoder()
    features["audio"] = audio_enc.encode(audio_path)
    del audio_enc

    from vie_gameemo.encoders.context_vit import ContextEncoder
    ctx_enc = ContextEncoder()
    if webcam_bbox is not None:
        from vie_gameemo.preprocess.face_crop import batch_extract_webcam_regions
        webcam_crops = batch_extract_webcam_regions(frame_paths, webcam_bbox)
        features["context"] = ctx_enc.encode(webcam_crops)
    else:
        features["context"] = ctx_enc.encode_from_paths(frame_paths)
    del ctx_enc

    from vie_gameemo.encoders.text_xlmr import XLMRTextEncoder
    text_enc = XLMRTextEncoder()
    features["text"] = text_enc.encode("")
    del text_enc

    from vie_gameemo.encoders.face_vit import FaceEncoder
    face_enc = FaceEncoder()
    face_tensor, has_face = face_enc.encode(frame_paths)
    features["face"] = face_tensor
    features["has_face"] = has_face
    del face_enc

    return features


def _forward(fusion, classifier, features: dict, device: torch.device) -> dict:
    """Run one forward pass, return label + confidence."""
    audio = features["audio"].unsqueeze(0).to(device)
    face = features["face"].unsqueeze(0).to(device)
    context = features["context"].unsqueeze(0).to(device)
    text = features["text"].unsqueeze(0).to(device)
    has_face = features.get("has_face")
    if has_face is not None:
        has_face = torch.tensor([[has_face]], dtype=torch.bool, device=device)

    with torch.no_grad():
        fused = fusion(audio, face, context, text, has_face=has_face)
        if isinstance(fused, tuple):
            fused = fused[0]
        logits = classifier(fused)
        probs = torch.softmax(logits, dim=-1)[0]

    pred_idx = int(probs.argmax().item())
    return {
        "label": _EMOTION_LABELS[pred_idx] if pred_idx < len(_EMOTION_LABELS) else str(pred_idx),
        "confidence": float(probs[pred_idx].item()),
        "class_scores": {_EMOTION_LABELS[i]: float(probs[i].item()) for i in range(len(_EMOTION_LABELS))},
        "fusion_emb": fused.cpu(),  # (1, T, 768) — for LLM modal adapter
    }


def _features_to_evidence(
    features: dict,
    prediction: dict,
    clip_path: Path,
    cfg: SimpleNamespace,
) -> dict:
    """Build LLM evidence dict.

    Uses annotation-free path (fusion_emb → ModalAdapter) when no text
    annotation is available; falls back to text evidence when annotation
    fields are present.
    """
    has_annotation = any(
        features.get(k) not in (None, "N/A", "")
        for k in ("face_aus", "visual_description", "audio_description")
    )
    fusion_emb = prediction.get("fusion_emb")

    if not has_annotation and fusion_emb is not None:
        return {
            "fusion_emb": fusion_emb,
            "label": prediction.get("label", "neutral"),
        }

    return {
        "face_aus": features.get("face_aus", "N/A"),
        "visual_objective": features.get("visual_description", "N/A"),
        "audio_tone": features.get("audio_description", "N/A"),
        "transcript": features.get("transcript", ""),
        "label": prediction.get("label", "neutral"),
    }
