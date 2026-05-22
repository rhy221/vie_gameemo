"""Stage 0: Crawl videos from URL list.

Downloads videos via yt-dlp, organizes by streamer/genre.

Usage:
    python scripts/stage0_crawl.py --config config.yaml
    python scripts/stage0_crawl.py --config config.yaml --source-list data/source_urls.txt --output data/raw_videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/stage0_crawl.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.data.crawler import crawl_videos
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0: crawl videos from URL list")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Main config file")
    parser.add_argument("--source-list", type=Path, default=None, help="URL list file (overrides config)")
    parser.add_argument("--output", type=Path, default=None, help="Output dir (overrides config)")
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Stop after downloading this many videos (for pilot runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    source_list = args.source_list or Path(cfg.crawl.source_list)
    output_dir = args.output or Path(cfg.paths.raw_videos)

    logger.info("Crawling videos from %s → %s", source_list, output_dir)
    paths = crawl_videos(
        source_list=source_list,
        output_dir=output_dir,
        video_format=cfg.crawl.video_format,
        rate_limit_kbps=cfg.crawl.rate_limit_kbps,
        retry_attempts=cfg.crawl.retry_attempts,
    )
    logger.info("Downloaded %d videos", len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
