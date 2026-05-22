"""Stage 0/1: Demux videos into audio + frames.

Extracts wav (16kHz mono) and frames (2fps) for each clip. Runs webcam
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

    detector = None
    bbox_map: dict[str, dict | None] = {}
    if not args.skip_webcam_detect:
        detector = WebcamDetector(
            min_detection_confidence=cfg.visual_encoder.webcam_detector.min_detection_confidence,
            sample_n_frames=cfg.visual_encoder.webcam_detector.sample_n_frames,
            dbscan_eps=cfg.visual_encoder.webcam_detector.clustering.eps,
            dbscan_min_samples=cfg.visual_encoder.webcam_detector.clustering.min_samples,
            stability_threshold=cfg.visual_encoder.webcam_detector.stability_threshold,
            edge_bias=cfg.visual_encoder.webcam_detector.edge_bias,
        )

    for video_path in video_paths:
        clip_id = video_path.stem
        logger.info("Processing %s", clip_id)

        # Audio
        extract_audio(
            video_path=video_path,
            output_path=audio_dir / f"{clip_id}.wav",
            sample_rate=cfg.preprocess.audio.sample_rate,
            channels=cfg.preprocess.audio.channels,
        )

        # Frames
        clip_frames_dir = ensure_dir(frames_dir / clip_id)
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

    if not args.skip_webcam_detect:
        write_json(bbox_map, webcam_bbox_file)
        logger.info("Saved webcam bboxes for %d clips → %s", len(bbox_map), webcam_bbox_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
