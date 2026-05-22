"""Single-clip or batch inference on new videos.

Usage:
    # Single clip
    python scripts/infer.py --config config.yaml --checkpoint outputs/checkpoints/best.pt \\
        --input clip.mp4 --output prediction.json

    # Batch (directory of clips)
    python scripts/infer.py --config config.yaml --checkpoint outputs/checkpoints/best.pt \\
        --input data/new_clips/ --output predictions.json --batch

    # Without LLM explanation (faster)
    python scripts/infer.py --config ... --checkpoint ... --input ... --no-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.inference.batch import batch_inference
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference on new video(s)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="Single clip or directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/results/predictions.json"))
    parser.add_argument("--batch", action="store_true",
                        help="Input is a directory; process all .mp4 files")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM explanation step (faster)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    if args.batch:
        clip_paths = sorted(args.input.glob("**/*.mp4"))
    else:
        if not args.input.exists():
            raise FileNotFoundError(f"Input not found: {args.input}")
        clip_paths = [args.input]

    logger.info("Running inference on %d clip(s)", len(clip_paths))
    output_path = batch_inference(
        clip_paths=clip_paths,
        checkpoint=args.checkpoint,
        cfg=cfg,
        output_json=args.output,
        include_llm_explanation=not args.no_llm,
        use_cached_features=False,  # raw inference
    )
    logger.info("Predictions saved → %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
