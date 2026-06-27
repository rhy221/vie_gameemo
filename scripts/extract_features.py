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
    relevant = {
        "audio": cfg.audio_encoder.model_name,
        "audio_tokens": cfg.audio_encoder.target_tokens,
        "face": cfg.visual_encoder.face_encoder.model_name,
        "context": cfg.visual_encoder.context_encoder.model_name,
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
        from vie_gameemo.encoders.audio_ast import ASTAudioEncoder
        logger.info("Loading audio encoder...")
        encoders["audio"] = ASTAudioEncoder(
            model_name=cfg.audio_encoder.model_name,
            target_tokens=cfg.audio_encoder.target_tokens,
            sample_rate=cfg.preprocess.audio.sample_rate,
            device=device,
        )

    if "face" in args.modalities:
        from vie_gameemo.encoders.face_vit import FaceEncoder
        logger.info("Loading face encoder...")
        encoders["face"] = FaceEncoder(
            model_name=cfg.visual_encoder.face_encoder.model_name,
            n_temporal_frames=cfg.visual_encoder.face_encoder.dual_view.temporal.n_frames,
            target_size=tuple(cfg.visual_encoder.face_encoder.target_size),
            device=device,
        )

    if "context" in args.modalities:
        from vie_gameemo.encoders.context_vit import ContextEncoder
        logger.info("Loading context encoder...")
        encoders["context"] = ContextEncoder(
            model_name=cfg.visual_encoder.context_encoder.model_name,
            n_frames=cfg.visual_encoder.context_encoder.n_frames,
            target_size=tuple(cfg.visual_encoder.context_encoder.target_size),
            temporal_pool=cfg.visual_encoder.context_encoder.temporal_pool,
            device=device,
        )

    if "text" in args.modalities:
        from vie_gameemo.encoders.text_xlmr import build_text_encoder
        logger.info("Loading text encoder (%s)...", getattr(cfg.text_encoder, "backend", getattr(cfg.text_encoder, "type", "xlmr")))
        enc = build_text_encoder(cfg.text_encoder)
        enc = enc.to(device)
        encoders["text"] = enc

    strategy = getattr(cfg.visual_encoder, "strategy", "dual_path")
    logger.info("Visual strategy: %s", strategy)

    # Load webcam bboxes from stage0_preprocess (separate from annotations)
    webcam_bboxes: dict[str, dict | None] = {}
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
                # Strategy A: full-frame → ViT-FER (no webcam crop)
                full_crops = [cv2.imread(str(fp)) for fp in frame_paths]
                full_crops = [c for c in full_crops if c is not None]
                feat, _ = encoders["face"].encode(full_crops if full_crops else None)
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
                # No webcam → full-frame fallback (ViT-FER tự xử lý)
                logger.info("No webcam for %s — face encoder uses full frames", clip_id)
                full_crops = [cv2.imread(str(fp)) for fp in frame_paths]
                full_crops = [c for c in full_crops if c is not None]
                feat, _ = encoders["face"].encode(full_crops if full_crops else None)
                features["face"] = feat.squeeze(0)

        if "context" in encoders:
            if strategy == "face_only":
                # Strategy B: no context, fill zeros
                T = 1 if cfg.visual_encoder.context_encoder.temporal_pool == "mean" else cfg.visual_encoder.context_encoder.n_frames
                features["context"] = torch.zeros(T, 768)
            elif strategy == "dual_path" and has_face and resolved_bbox is not None:
                # Strategy C: webcam region crops for context
                from vie_gameemo.preprocess.face_crop import batch_extract_webcam_regions
                webcam_crops = batch_extract_webcam_regions(
                    frame_paths, resolved_bbox,
                    target_size=tuple(cfg.visual_encoder.context_encoder.target_size),
                )
                features["context"] = encoders["context"].encode(webcam_crops).squeeze(0)
            else:
                # full_frame OR dual_path fallback: full frames
                if strategy == "dual_path":
                    logger.warning("Webcam not detected for %s — context encoder fallback to full frames", clip_id)
                features["context"] = encoders["context"].encode_from_paths(frame_paths).squeeze(0)

        if "text" in encoders:
            transcript = ann.transcript or ""
            features["text"] = encoders["text"].encode(transcript).squeeze(0)

        features["has_face"] = torch.tensor([has_face], dtype=torch.bool)

        cache_features(clip_id, features, features_dir, overwrite=True,
                       config_hash=cfg_hash)
        n_done += 1
        logger.info("[%d/%d] Cached: %s", n_done, len(annotation_files), clip_id)

    logger.info("Feature extraction complete. %d clips cached.", n_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
