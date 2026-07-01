"""Stage 2: Extract and cache features from frozen encoders.

Runs the 4 encoders (AST audio, ViT-FER face, ViT-ImageNet context, XLM-R text)
on all annotated clips, caches outputs as .pt files under data/features/.

Once cached, training reads features from disk and skips encoders entirely
(big speedup on repeated training runs).

Usage:
    python scripts/extract_features.py --config config.yaml
    python scripts/extract_features.py --config config.yaml --modalities audio face
    python scripts/extract_features.py --config config.yaml --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from vie_gameemo.data.feature_cache import cache_features, is_cache_valid
from vie_gameemo.data.schemas import Annotation
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.io import ensure_dir
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: extract + cache features")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--modalities", nargs="+",
        default=["audio", "face", "context", "text"],
        choices=["audio", "face", "context", "text"],
        help="Which modalities to extract",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N clips")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _config_hash(cfg) -> str:
    """Compute a hash of the encoder config for cache invalidation."""
    ctx_cfg = cfg.visual_encoder.context_encoder
    ctx_type = getattr(ctx_cfg, "type", "vit_imagenet")
    # Distinguish pose vs vit_imagenet; pose has no model_name
    ctx_key = (
        getattr(ctx_cfg, "model_name", "")
        if ctx_type != "pose"
        else f"pose:{getattr(ctx_cfg, 'pose_backend', 'mediapipe')}"
    )
    relevant = {
        "audio": cfg.audio_encoder.model_name,
        "audio_tokens": cfg.audio_encoder.target_tokens,
        "face": cfg.visual_encoder.face_encoder.model_name,
        "context": ctx_key,
        "text": getattr(cfg.text_encoder, "model", getattr(cfg.text_encoder, "model_name", "unknown")),
        "text_backend": getattr(cfg.text_encoder, "backend", getattr(cfg.text_encoder, "type", "xlmr")),
    }
    return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    annotations_dir = Path(cfg.paths.annotations)
    features_dir = ensure_dir(Path(cfg.paths.features))
    processed_dir = Path(cfg.paths.processed)

    annotation_files = sorted(annotations_dir.glob("*.json"))
    if args.limit:
        annotation_files = annotation_files[:args.limit]
    logger.info("Extracting features for %d clips → %s", len(annotation_files), features_dir)

    device = torch.device(
        cfg.compute.device if cfg.compute.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)
    cfg_hash = _config_hash(cfg)

    # Load encoders lazily per modality
    encoders: dict = {}

    if "audio" in args.modalities:
        from vie_gameemo.encoders.audio_whisper import WhisperAudioEncoder
        logger.info("Loading audio encoder...")
        encoders["audio"] = WhisperAudioEncoder(
            model_name=cfg.audio_encoder.model_name,
            target_tokens=cfg.audio_encoder.target_tokens,
            sample_rate=cfg.preprocess.audio.sample_rate,
            device=device,
        )

    if "face" in args.modalities:
        from vie_gameemo.encoders.face_vit import FaceEncoder
        logger.info("Loading face encoder...")
        _face_cfg = cfg.visual_encoder.face_encoder
        _tv_cfg = _face_cfg.tri_view.temporal
        encoders["face"] = FaceEncoder(
            model_name=_face_cfg.model_name,
            n_temporal_frames=_tv_cfg.n_frames,
            spatial_pool=tuple(_tv_cfg.spatial_pool),
            temporal_spatial_pool=tuple(getattr(_tv_cfg, "temporal_spatial_pool", [2, 2])),
            pool_method=_tv_cfg.pool_method,
            target_size=tuple(_face_cfg.target_size),
            use_temporal_3d_pool=getattr(_tv_cfg, "use_temporal_3d_pool", False),
            temporal_3d_pool=tuple(getattr(_tv_cfg, "temporal_3d_pool", [4, 4, 4])),
            device=device,
        )

    if "context" in args.modalities:
        from vie_gameemo.encoders import get_context_encoder
        ctx_enc_type = getattr(cfg.visual_encoder.context_encoder, "type", "vit_imagenet")
        logger.info("Loading context encoder (type=%s)...", ctx_enc_type)
        encoders["context"] = get_context_encoder(cfg).to(device)

    if "text" in args.modalities:
        from vie_gameemo.encoders.text_xlmr import build_text_encoder
        logger.info("Loading text encoder (%s)...", getattr(cfg.text_encoder, "backend", getattr(cfg.text_encoder, "type", "xlmr")))
        enc = build_text_encoder(cfg.text_encoder)
        enc = enc.to(device)
        encoders["text"] = enc

    strategy = getattr(cfg.visual_encoder, "strategy", "dual_path")
    logger.info("Visual strategy: %s", strategy)

    # Load webcam bboxes (only needed for face_only / dual_path strategies)
    webcam_bboxes: dict[str, dict | None] = {}
    if strategy != "full_frame":
        webcam_bbox_file = Path(cfg.paths.processed) / "webcam_bboxes.json"
        if webcam_bbox_file.exists():
            import json as _json
            webcam_bboxes = _json.loads(webcam_bbox_file.read_text(encoding="utf-8"))
            logger.info("Loaded %d webcam bboxes from %s", len(webcam_bboxes), webcam_bbox_file)

    n_done = 0
    for ann_file in annotation_files:
        clip_id = ann_file.stem
        pt_path = features_dir / f"{clip_id}.pt"

        if not args.overwrite and pt_path.exists():
            logger.debug("Cache exists (skip): %s", clip_id)
            n_done += 1
            continue

        try:
            ann = Annotation.load(ann_file)
        except Exception as exc:
            logger.warning("Failed to load annotation %s: %s", ann_file, exc)
            continue

        features: dict[str, torch.Tensor] = {}
        # Resolve webcam bbox: annotation first, then webcam_bboxes.json fallback
        resolved_bbox = ann.webcam_bbox
        if resolved_bbox is None and clip_id in webcam_bboxes and webcam_bboxes[clip_id] is not None:
            bbox_dict = webcam_bboxes[clip_id]
            from vie_gameemo.preprocess.webcam_detector import WebcamBBox as DetectorBBox
            resolved_bbox = DetectorBBox(
                xmin=bbox_dict["xmin"], ymin=bbox_dict["ymin"],
                width=bbox_dict["width"], height=bbox_dict["height"],
                stability_score=bbox_dict.get("stability_score", 0.0),
                edge_distance=bbox_dict.get("edge_distance", 0.0),
            )
        has_face = resolved_bbox is not None
        frame_dir = Path(cfg.paths.frames) / clip_id
        frame_paths = sorted(frame_dir.glob("frame_*.jpg")) if frame_dir.exists() else []

        if "audio" in encoders:
            audio_path = Path(cfg.paths.audios) / f"{clip_id}.wav"
            if audio_path.exists():
                features["audio"] = encoders["audio"].encode(audio_path).squeeze(0)
            else:
                logger.warning("Audio file missing: %s", audio_path)
                features["audio"] = torch.zeros(cfg.audio_encoder.target_tokens, 768)

        import cv2
        import numpy as np

        if "face" in encoders:
            if strategy == "full_frame":
                # Strategy A: no webcam detection, but still crop face via MediaPipe
                from vie_gameemo.preprocess.face_crop import _tight_face_crop
                crops = []
                for fp in frame_paths:
                    frame = cv2.imread(str(fp))
                    if frame is not None:
                        crops.append(_tight_face_crop(frame, fallback=frame))
                feat, _ = encoders["face"].encode(crops if crops else None)
                features["face"] = feat.squeeze(0)
            elif has_face and resolved_bbox is not None and frame_paths:
                # Strategy B/C: crop face on-the-fly từ bbox + frames
                from vie_gameemo.preprocess.face_crop import extract_streamer_face
                margin = getattr(cfg.visual_encoder.face_encoder, "crop_margin", 0.2)
                crops = []
                for fp in frame_paths:
                    frame = cv2.imread(str(fp))
                    if frame is not None:
                        crops.append(extract_streamer_face(frame, resolved_bbox, margin=margin))
                feat, _ = encoders["face"].encode(crops if crops else None)
                features["face"] = feat.squeeze(0)
            else:
                # No webcam bbox → detect face in full-frame via MediaPipe, crop nếu tìm thấy
                from vie_gameemo.preprocess.face_crop import _tight_face_crop
                logger.info("No webcam for %s — detecting face in full frames", clip_id)
                crops = []
                for fp in frame_paths:
                    frame = cv2.imread(str(fp))
                    if frame is not None:
                        cropped = _tight_face_crop(frame, fallback=frame)
                        crops.append(cropped)
                feat, _ = encoders["face"].encode(crops if crops else None)
                features["face"] = feat.squeeze(0)

        if "context" in encoders:
            # Always extract real context features regardless of strategy.
            # Zeroing for face_only is applied at Dataset load time, not here,
            # so the cache remains strategy-agnostic and can be reused across runs.
            ctx_enc_type = getattr(cfg.visual_encoder.context_encoder, "type", "vit_imagenet")
            if strategy == "dual_path" and has_face and resolved_bbox is not None:
                from vie_gameemo.preprocess.face_crop import batch_extract_webcam_regions
                if ctx_enc_type == "pose":
                    # Pose branch: crop webcam WITHOUT resize (full native resolution for
                    # better keypoint detection). Same webcam region as ViT for ablation
                    # fairness (P0.0-fix.1). kps saved to shared cache (P0.0-fix.2).
                    webcam_crops_raw = batch_extract_webcam_regions(
                        frame_paths, resolved_bbox, margin=0.1, resize=False,
                    )
                    kps_cache_path = (
                        Path(cfg.paths.cache) / "pose_kinematics" / f"{clip_id}_kps.npy"
                    )
                    features["context"] = encoders["context"].encode(
                        webcam_crops_raw, kps_cache_path=kps_cache_path,
                    ).squeeze(0)
                else:
                    # ViT branch: crop + resize to 224×224 as before
                    webcam_crops = batch_extract_webcam_regions(
                        frame_paths, resolved_bbox,
                        target_size=tuple(cfg.visual_encoder.context_encoder.target_size),
                    )
                    features["context"] = encoders["context"].encode(webcam_crops).squeeze(0)
            else:
                # full_frame, face_only, or dual_path fallback: full frames
                if strategy == "dual_path":
                    logger.warning("Webcam not detected for %s — context encoder fallback to full frames", clip_id)
                features["context"] = encoders["context"].encode_from_paths(frame_paths).squeeze(0)

        if "text" in encoders:
            transcript = ann.transcript or ""
            features["text"] = encoders["text"].encode(transcript).squeeze(0)

        features["has_face"] = torch.tensor([has_face], dtype=torch.bool)

        # Merge with existing cache if only extracting a subset of modalities
        pt_path = features_dir / f"{clip_id}.pt"
        if pt_path.exists() and set(args.modalities) != {"audio", "face", "context", "text"}:
            existing = torch.load(pt_path, map_location="cpu", weights_only=False)
            existing.update(features)
            features = existing

        cache_features(clip_id, features, features_dir, overwrite=True,
                       config_hash=cfg_hash)
        n_done += 1
        logger.info("[%d/%d] Cached: %s", n_done, len(annotation_files), clip_id)

    logger.info("Feature extraction complete. %d clips cached.", n_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
