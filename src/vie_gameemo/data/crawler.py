"""Video crawler using yt-dlp.

Downloads videos from a URL list, organizes them under data/raw_videos/ by
streamer/genre. Resilient to network errors with retries.
"""

import hashlib
import logging
from pathlib import Path

from vie_gameemo.data.schemas import GameGenre
from vie_gameemo.utils.io import read_json, write_json

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"


def crawl_videos(
    source_list: Path,
    output_dir: Path,
    video_format: str = "best[height<=720]",
    rate_limit_kbps: int = 2000,
    retry_attempts: int = 3,
) -> list[Path]:
    """Download videos from a URL list.

    Args:
        source_list: Text file with one URL per line. Optionally TSV with
            columns: url, streamer, genre.
        output_dir: Where to save videos. Structured as output_dir/streamer/.
        video_format: yt-dlp format selector.
        rate_limit_kbps: Bandwidth cap (be polite to source servers).
        retry_attempts: Retries per video on transient failures.

    Returns:
        List of paths to downloaded video files.

    Raises:
        FileNotFoundError: If source_list does not exist.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError("yt-dlp not installed. Run: pip install yt-dlp") from e

    if not source_list.exists():
        raise FileNotFoundError(f"Source list not found: {source_list}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / _MANIFEST_FILENAME
    manifest: dict = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)

    entries = parse_source_list(source_list)
    downloaded: list[Path] = []

    rate_limit_bytes = rate_limit_kbps * 1024 if rate_limit_kbps else None

    for entry in entries:
        url = entry["url"]
        url_hash = _url_hash(url)
        if url_hash in manifest and Path(manifest[url_hash]["path"]).exists():
            logger.info("Already downloaded (skip): %s", url)
            downloaded.append(Path(manifest[url_hash]["path"]))
            continue

        streamer = entry.get("streamer", "unknown")
        save_dir = output_dir / streamer
        save_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts: dict = {
            "format": video_format,
            "outtmpl": str(save_dir / "%(id)s.%(ext)s"),
            "retries": retry_attempts,
            "quiet": True,
            "no_warnings": True,
        }
        if rate_limit_bytes:
            ydl_opts["ratelimit"] = rate_limit_bytes

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = Path(ydl.prepare_filename(info))
                downloaded.append(video_path)
                manifest[url_hash] = {
                    "url": url,
                    "streamer": streamer,
                    "genre": entry.get("genre", "unknown"),
                    "path": str(video_path),
                    "title": info.get("title", ""),
                }
                write_json(manifest, manifest_path)
                logger.info("Downloaded: %s → %s", url, video_path)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", url, exc)

    return downloaded


def parse_source_list(source_list: Path) -> list[dict]:
    """Parse the source list file.

    Supports:
        - Plain text (one URL per line)
        - TSV with header: url, streamer, genre

    Args:
        source_list: Path to source list file.

    Returns:
        List of dicts with keys: url, streamer (optional), genre (optional).
    """
    lines = source_list.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    if not lines:
        return []

    if "\t" in lines[0] and "url" in lines[0].lower():
        headers = [h.strip() for h in lines[0].split("\t")]
        entries = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split("\t")]
            entries.append(dict(zip(headers, parts)))
        return entries

    return [{"url": ln} for ln in lines]


def _url_hash(url: str) -> str:
    """Short hash of a URL for deduplication."""
    return hashlib.md5(url.encode()).hexdigest()[:12]
