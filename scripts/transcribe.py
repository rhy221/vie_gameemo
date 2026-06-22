"""Transcribe audio files and update annotation JSONs with transcript + language metadata.

Runs Whisper ASR on preprocessed audio (data/processed/audios/*.wav),
writes transcript back into each annotation JSON.

Usage:
    python scripts/transcribe.py --config config.yaml
    python scripts/transcribe.py --config config.yaml --backend phowhisper
    python scripts/transcribe.py --config config.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.data.annotator.whisper_asr import build_asr, transcribe_clip
from vie_gameemo.data.schemas import Annotation
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio → update annotations")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--backend", type=str, default=None, help="Override ASR backend (whisper|phowhisper)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N clips")
    parser.add_argument("--overwrite", action="store_true", help="Re-transcribe clips that already have transcript")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    annotations_dir = Path(cfg.paths.annotations)
    audio_dir = Path(cfg.paths.audios)

    annotation_files = sorted(annotations_dir.glob("*.json"))
    if args.limit:
        annotation_files = annotation_files[:args.limit]

    if not annotation_files:
        logger.error("No annotation files in %s", annotations_dir)
        return 1

    logger.info("Found %d annotations, audio dir: %s", len(annotation_files), audio_dir)

    asr_cfg = cfg.annotation.asr
    if args.backend:
        asr_cfg.backend = args.backend

    asr_inst, bartpho_inst = build_asr(asr_cfg)
    asr_inst.load()
    if bartpho_inst is not None:
        bartpho_inst.load()

    transcribed = 0
    skipped = 0
    missing = 0

    try:
        for ann_file in annotation_files:
            ann = Annotation.load(ann_file)

            if ann.transcript.strip() and not args.overwrite:
                skipped += 1
                continue

            audio_path = audio_dir / f"{ann.clip_id}.wav"
            if not audio_path.exists():
                missing += 1
                continue

            try:
                result = transcribe_clip(
                    asr_inst, bartpho_inst, audio_path,
                    source_language=ann.source_language,
                    asr_cfg=asr_cfg,
                )
                ann.transcript = result.text
                ann.asr_detected_language = result.asr_detected_language
                ann.text_detected_language = result.text_detected_language
                ann.language_detect_confidence = result.language_detect_confidence
                ann.language_mismatch = result.language_mismatch
                ann.save(ann_file)
                transcribed += 1

                if transcribed % 50 == 0:
                    logger.info("  [%d/%d] transcribed", transcribed, len(annotation_files))

            except Exception as exc:
                logger.warning("ASR failed for %s: %s", ann.clip_id, exc)

    finally:
        asr_inst.unload()
        if bartpho_inst is not None:
            bartpho_inst.unload()

    logger.info("Done: %d transcribed, %d skipped (had transcript), %d missing audio",
                transcribed, skipped, missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
