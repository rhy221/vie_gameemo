"""Stage 0/1: Demux videos into audio + frames.

Extracts wav (16kHz mono) and frames (4fps) for each clip. Runs webcam
detection and caches the bbox per clip. Outputs cached under data/processed/.

This script is part of the Stage 0 group (data prep) and runs INDEPENDENTLY
from the model pipeline.

Usage:
    python scripts/stage0_preprocess.py --config config.yaml
    python scripts/stage0_preprocess.py --config config.yaml --videos-dir data/raw_videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.preprocess.demux import extract_audio, extract_frames
from vie_gameemo.preprocess.webcam_detector import WebcamDetector
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.io import ensure_dir, write_json
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0/1: demux clips + webcam detection")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--videos-dir", type=Path, default=None, help="Input videos directory")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N clips")
    parser.add_argument("--skip-webcam-detect", action="store_true", help="Skip webcam detection step")
    parser.add_argument("--webcam-only", action="store_true", help="Only re-run webcam detection (skip audio/frames)")
    parser.add_argument("--resume", action="store_true", help="Skip clips that already have audio + frames")
    parser.add_argument("--overwrite", action="store_true", help="Re-process all clips even if output exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    videos_dir = args.videos_dir or Path(cfg.paths.raw_videos)
    audio_dir = ensure_dir(Path(cfg.paths.audios))
    frames_dir = ensure_dir(Path(cfg.paths.frames))
    webcam_bbox_file = ensure_dir(Path(cfg.paths.processed)) / "webcam_bboxes.json"

    video_paths = sorted(videos_dir.glob("**/*.mp4"))
    if args.limit:
        video_paths = video_paths[: args.limit]
    logger.info("Found %d videos to process", len(video_paths))

    strategy = getattr(cfg.visual_encoder, "strategy", "dual_path")
    skip_webcam = args.skip_webcam_detect or (strategy == "full_frame" and not args.webcam_only)
    if skip_webcam:
        logger.info("Webcam detection skipped (strategy=%s, flag=%s)", strategy, args.skip_webcam_detect)
    if args.webcam_only:
        logger.info("Webcam-only mode: skipping audio/frames extraction")

    detector = None
    bbox_map: dict[str, dict | None] = {}
    if not skip_webcam:
        wc = cfg.visual_encoder.webcam_detector
        backend = getattr(wc, "backend", "mediapipe")
        owlv2_cfg = getattr(wc, "owlv2", None)
        yolo_cfg = getattr(wc, "yolo", None)
        detector = WebcamDetector(
            backend=backend,
            min_detection_confidence=wc.min_detection_confidence,
            sample_n_frames=wc.sample_n_frames,
            dbscan_eps=wc.clustering.eps,
            dbscan_min_samples=wc.clustering.min_samples,
            stability_threshold=wc.stability_threshold,
            edge_bias=wc.edge_bias,
            owlv2_model=getattr(owlv2_cfg, "model", "google/owlv2-base-patch16-finetuned") if owlv2_cfg else "google/owlv2-base-patch16-finetuned",
            owlv2_prompt=getattr(owlv2_cfg, "prompt", "facecam overlay") if owlv2_cfg else "facecam overlay",
            yolo_model=getattr(yolo_cfg, "model", "yolo11n.pt") if yolo_cfg else "yolo11n.pt",
            yolo_classes=list(getattr(yolo_cfg, "classes", [0])) if yolo_cfg else [0],
        )
        logger.info("Webcam detector backend: %s", backend)

    # Load existing webcam bboxes for resume mode
    if not skip_webcam and args.resume and webcam_bbox_file.exists():
        import json
        bbox_map = json.loads(webcam_bbox_file.read_text(encoding="utf-8"))
        logger.info("Resumed %d existing webcam bboxes", len(bbox_map))

    n_skipped = 0
    for video_path in video_paths:
        clip_id = video_path.stem
        audio_out = audio_dir / f"{clip_id}.wav"
        clip_frames_dir = frames_dir / clip_id

        # Resume: skip if audio + frames already exist
        if args.resume and not args.overwrite and not args.webcam_only:
            has_audio = audio_out.exists()
            has_frames = clip_frames_dir.exists() and any(clip_frames_dir.glob("frame_*.jpg"))
            has_bbox = skip_webcam or clip_id in bbox_map
            if has_audio and has_frames and has_bbox:
                n_skipped += 1
                continue

        logger.info("Processing %s", clip_id)

        if not args.webcam_only:
            # Audio
            extract_audio(
                video_path=video_path,
                output_path=audio_out,
                sample_rate=cfg.preprocess.audio.sample_rate,
                channels=cfg.preprocess.audio.channels,
            )

            # Frames
            clip_frames_dir = ensure_dir(clip_frames_dir)
            extract_frames(
                video_path=video_path,
                output_dir=clip_frames_dir,
                fps=cfg.preprocess.frames.fps,
                quality=cfg.preprocess.frames.quality,
            )

        # Webcam detection
        if detector is not None:
            bbox = detector.detect_webcam_region(video_path)
            bbox_map[clip_id] = bbox.__dict__ if bbox else None

    if n_skipped:
        logger.info("Skipped %d already-processed clips (resume mode)", n_skipped)

    if not skip_webcam:
        write_json(bbox_map, webcam_bbox_file)
        logger.info("Saved webcam bboxes for %d clips → %s", len(bbox_map), webcam_bbox_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
