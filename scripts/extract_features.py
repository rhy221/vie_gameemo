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
from vie_gameemo.data.feature_extraction import (
    build_augment_fns,
    extract_clip_features,
    index_videos,
    resolve_webcam_bbox,
)
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
    parser.add_argument(
        "--videos-dir", type=Path, default=None,
        help="Source videos dir (default: cfg.paths.raw_videos). Used as audio "
             "fallback: if a clip's .wav is missing, audio is read straight from "
             "the source .mp4 (librosa decodes it via ffmpeg).",
    )
    parser.add_argument(
        "--augment-copies", type=int, default=0,
        help="Precomputed raw-augmentation mode (config: augment.raw.*): for "
             "each clip, additionally cache N randomly-augmented variants "
             "({clip_id}_aug{i}.pt, color/crop jitter for face+context, "
             "pitch-shift/time-stretch/noise/SpecAugment for audio) alongside "
             "the original. VieGameEmoDataset randomly picks among all cached "
             "variants each epoch when raw_augment.mode='precomputed'. 0 "
             "(default) caches only the original, unaugmented features.",
    )
    return parser.parse_args()


def _config_hash(cfg) -> str:
    """Compute a hash of the encoder config for cache invalidation."""
    ctx_cfg = cfg.visual_encoder.context_encoder
    ctx_type = getattr(ctx_cfg, "type", "vit_imagenet")
    from vie_gameemo.encoders import resolve_audio_model_name, resolve_context_vit_model_name

    # Distinguish pose vs vit_imagenet; pose has no model_name.
    # For vit_imagenet, resolve via backend/models (not the flat model_name
    # field alone) so switching context_encoder.backend (e.g. vit → eva_vit_b)
    # invalidates the cache even if the flat model_name field is untouched.
    if ctx_type == "pose":
        ctx_key = f"pose:{getattr(ctx_cfg, 'pose_backend', 'mediapipe')}"
    else:
        ctx_backend = getattr(ctx_cfg, "backend", "vit")
        ctx_key = f"{ctx_backend}:{resolve_context_vit_model_name(ctx_cfg, ctx_backend)}"

    relevant = {
        "audio_type": getattr(cfg.audio_encoder, "type", "whisper"),
        "audio": resolve_audio_model_name(cfg),
        "audio_tokens": cfg.audio_encoder.target_tokens,
        "audio_d_out": getattr(cfg.audio_encoder, "d_out", None),
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

    video_index: dict[str, Path] = {}
    if "audio" in args.modalities:
        from vie_gameemo.encoders import get_audio_encoder
        logger.info("Loading audio encoder (type=%s)...", getattr(cfg.audio_encoder, "type", "whisper"))
        encoders["audio"] = get_audio_encoder(cfg, device=device)
        videos_dir = args.videos_dir or Path(cfg.paths.raw_videos)
        video_index = index_videos(videos_dir)
        logger.info("Indexed %d source videos for audio fallback (%s)",
                    len(video_index), videos_dir)

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

        # Resolve webcam bbox: annotation first, then webcam_bboxes.json fallback
        resolved_bbox = resolve_webcam_bbox(ann, clip_id, webcam_bboxes)
        has_face = resolved_bbox is not None
        frame_dir = Path(cfg.paths.frames) / clip_id
        frame_paths = sorted(frame_dir.glob("frame_*.jpg")) if frame_dir.exists() else []

        features = extract_clip_features(
            cfg, encoders, clip_id, ann, resolved_bbox, has_face, frame_paths,
            strategy, video_index, logger,
        )

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

        if args.augment_copies > 0:
            image_aug_fn, audio_aug_fn = build_augment_fns(cfg)
            for i in range(args.augment_copies):
                aug_id = f"{clip_id}_aug{i}"
                aug_pt_path = features_dir / f"{aug_id}.pt"
                if not args.overwrite and aug_pt_path.exists():
                    continue
                aug_features = extract_clip_features(
                    cfg, encoders, clip_id, ann, resolved_bbox, has_face, frame_paths,
                    strategy, video_index, logger,
                    image_aug=image_aug_fn, audio_aug=audio_aug_fn,
                )
                cache_features(aug_id, aug_features, features_dir, overwrite=True,
                               config_hash=cfg_hash)
                logger.info("  + augmented variant %d/%d cached: %s", i + 1, args.augment_copies, aug_id)

    logger.info("Feature extraction complete. %d clips cached.", n_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
