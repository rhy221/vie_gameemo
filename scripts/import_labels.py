"""Import labels from flat JSON files into Annotation JSONs + split manifest.

Expected input layout:
    data/
    ├── raw_videos/
    │   ├── train/    ← .mp4 clips
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train.json   ← [{id, video, choice, confidence}, ...]
        ├── val.json
        └── test.json

Output:
    data/
    ├── annotations/     ← one Annotation JSON per clip
    └── splits.json      ← {clip_id: "train"|"val"|"test"}

Usage:
    python scripts/import_labels.py
    python scripts/import_labels.py --data-root data --labels-dir data/labels
    python scripts/import_labels.py --splits train val test
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.data.schemas import Annotation, EmotionLabel, GameGenre

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

VALID_LABELS = {e.value for e in EmotionLabel}


def load_label_json(path: Path) -> list[dict]:
    """Load a label JSON file.

    Accepts two formats:
        1. [{id, video, choice, confidence}, ...]       ← your format
        2. [{clip_id, emotion_label, ...}, ...]          ← already converted
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__} in {path}")

    return data


def entry_to_annotation(entry: dict, split: str, video_dir: Path) -> Annotation:
    """Convert a flat label entry to an Annotation object."""
    video_name = entry.get("video", "")
    clip_id = Path(video_name).stem
    choice = entry.get("choice", "neutral")

    if choice not in VALID_LABELS:
        logger.warning("Unknown label '%s' for %s, defaulting to 'neutral'", choice, clip_id)
        choice = "neutral"

    genre = _infer_genre_from_filename(clip_id)

    return Annotation(
        clip_id=clip_id,
        emotion_label=EmotionLabel(choice),
        genre=GameGenre(genre),
        face_aus={},
        peak_frame_idx=0,
        visual_objective_desc="",
        audio_tone_desc="",
        transcript="",
        reasoning="",
        annotators=[],
        human_verified=False,
        code_switching_ratio=0.0,
        source_language="vi",
        created_at=datetime.now(timezone.utc),
    )


def _infer_genre_from_filename(clip_id: str) -> str:
    """Best-effort genre inference from filename. Defaults to 'moba'."""
    name = clip_id.lower()
    for genre in GameGenre:
        if genre.value in name:
            return genre.value
    return GameGenre.MOBA.value


def import_split(
    label_file: Path,
    split_name: str,
    video_dir: Path,
    annotations_dir: Path,
) -> list[tuple[str, str]]:
    """Import one split's labels.

    Returns:
        List of (clip_id, split_name) for the manifest.
    """
    entries = load_label_json(label_file)
    logger.info("Importing %s: %d entries from %s", split_name, len(entries), label_file)

    manifest_entries = []
    label_counts: Counter = Counter()

    for entry in entries:
        ann = entry_to_annotation(entry, split_name, video_dir)
        ann_path = annotations_dir / f"{ann.clip_id}.json"
        ann.save(ann_path)
        manifest_entries.append((ann.clip_id, split_name))
        label_counts[ann.emotion_label.value] += 1

    logger.info("  Labels: %s", dict(sorted(label_counts.items())))
    return manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Import flat label JSONs into Annotation format")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="Directory with train.json/val.json/test.json (default: data/labels)")
    parser.add_argument("--videos-dir", type=Path, default=None,
                        help="Directory with train/val/test subfolders (default: data/raw_videos)")
    parser.add_argument("--annotations-dir", type=Path, default=None,
                        help="Output annotations directory (default: data/annotations)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Split names to import")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Output split manifest path (default: data/splits.json)")
    args = parser.parse_args()

    labels_dir = args.labels_dir or args.data_root / "labels"
    videos_dir = args.videos_dir or args.data_root / "raw_videos"
    annotations_dir = args.annotations_dir or args.data_root / "annotations"
    split_manifest_path = args.split_manifest or args.data_root / "splits.json"

    annotations_dir.mkdir(parents=True, exist_ok=True)

    all_manifest: dict[str, str] = {}
    total = 0

    for split_name in args.splits:
        label_file = labels_dir / f"{split_name}.json"
        if not label_file.exists():
            logger.warning("Label file not found: %s — skipping", label_file)
            continue

        video_dir = videos_dir / split_name
        entries = import_split(label_file, split_name, video_dir, annotations_dir)
        for clip_id, split in entries:
            all_manifest[clip_id] = split
        total += len(entries)

    # Save split manifest
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 50)
    logger.info("IMPORT COMPLETE")
    logger.info("  Total clips: %d", total)
    logger.info("  Annotations: %s (%d files)", annotations_dir, total)
    logger.info("  Split manifest: %s", split_manifest_path)
    for split_name in args.splits:
        count = sum(1 for v in all_manifest.values() if v == split_name)
        logger.info("    %s: %d clips", split_name, count)
    logger.info("=" * 50)

    # Verify videos exist
    missing = 0
    for clip_id, split in all_manifest.items():
        video_dir = videos_dir / split
        candidates = list(video_dir.glob(f"{clip_id}.*")) if video_dir.exists() else []
        if not candidates:
            missing += 1
    if missing:
        logger.warning("  %d clips have no matching video file in raw_videos/", missing)
    else:
        logger.info("  All clips have matching video files")


if __name__ == "__main__":
    main()
