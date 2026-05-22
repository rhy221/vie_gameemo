"""Stage 0: Multi-agent annotation.

Runs the multi-agent pipeline (OpenFace + Qwen-VL + Qwen-Audio + Whisper +
Consolidator) over preprocessed clips. Outputs annotation JSON files.

Heavy compute — run on A100 if available, or Colab T4 with smaller models.
This script is INDEPENDENT from the model training pipeline.

Resume support: clips with existing annotation files are skipped (toggle with --no-resume).

Usage:
    # Pilot run on 20 clips
    python scripts/stage0_annotate.py --config config.yaml --labels-csv data/labels.csv --limit 20

    # Full run
    python scripts/stage0_annotate.py --config config.yaml --labels-csv data/labels.csv

    # Force re-annotate (overwrite)
    python scripts/stage0_annotate.py --config config.yaml --labels-csv data/labels.csv --no-resume
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.data.annotator.pipeline import annotate_batch
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.io import ensure_dir
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0: multi-agent annotation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--labels-csv", type=Path, required=True,
        help="CSV with columns: clip_id, emotion_label. From manual pilot annotation.",
    )
    parser.add_argument("--videos-dir", type=Path, default=None,
                        help="Directory containing video clips (overrides config)")
    parser.add_argument("--limit", type=int, default=None, help="Annotate only first N clips")
    parser.add_argument("--no-resume", action="store_true", help="Re-annotate even if output exists")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    return parser.parse_args()


def parse_labels_csv(
    csv_path: Path,
    videos_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Parse labels CSV and resolve clip paths.

    Args:
        csv_path: CSV with columns: clip_id, emotion_label.
        videos_dir: Directory containing video files.

    Returns:
        Tuple of (clip_paths, emotion_labels).

    Raises:
        FileNotFoundError: If CSV not found.
        ValueError: If CSV format is invalid.
    """
    clip_paths: list[Path] = []
    emotion_labels: list[str] = []

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"clip_id", "emotion_label"}
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Labels CSV must have columns: {required_cols}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            clip_id = row["clip_id"].strip()
            label = row["emotion_label"].strip().lower()

            # Try common video extensions
            video_path: Path | None = None
            for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
                candidate = videos_dir / f"{clip_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break

            if video_path is None:
                # Also check subdirectories
                matches = list(videos_dir.rglob(f"{clip_id}.*"))
                video_path = matches[0] if matches else None

            if video_path is None:
                raise FileNotFoundError(
                    f"No video file found for clip_id={clip_id!r} in {videos_dir}"
                )
            clip_paths.append(video_path)
            emotion_labels.append(label)

    return clip_paths, emotion_labels


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    if not args.labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels_csv}")

    videos_dir = args.videos_dir or Path(cfg.paths.raw_videos)
    clip_paths, emotion_labels = parse_labels_csv(args.labels_csv, videos_dir)

    if args.limit:
        clip_paths = clip_paths[: args.limit]
        emotion_labels = emotion_labels[: args.limit]

    output_dir = ensure_dir(Path(cfg.paths.annotations))
    logger.info("Annotating %d clips → %s", len(clip_paths), output_dir)

    annotation_paths = annotate_batch(
        clip_paths=clip_paths,
        emotion_labels=emotion_labels,
        output_dir=output_dir,
        cfg=cfg.annotation,
        resume=not args.no_resume,
    )
    logger.info("Wrote %d annotation files", len(annotation_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
