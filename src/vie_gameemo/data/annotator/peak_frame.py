"""Peak frame detection: find the frame with highest emotional expression intensity.

Used as anchor for visual-objective description (Cvod) in the multi-agent
pipeline. Heuristic: emotion often peaks at a specific moment within a clip.
"""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def detect_peak_frame(
    video_path: Path,
    au_intensities: dict[int, list[float]] | None = None,
    method: str = "max_au_intensity",
) -> tuple[int, float]:
    """Detect the peak emotional frame in a video.

    Args:
        video_path: Input video.
        au_intensities: Pre-extracted AU intensities (required for 'max_au_intensity').
        method: "max_au_intensity" | "middle".

    Returns:
        Tuple of (frame_index, intensity_score).

    Raises:
        ValueError: If method unknown or AU data missing when required.
    """
    if method == "max_au_intensity":
        if not au_intensities:
            raise ValueError("au_intensities required for method='max_au_intensity'")
        aggregated = _aggregate_aus(au_intensities)
        if not aggregated:
            return 0, 0.0
        peak_idx = int(max(range(len(aggregated)), key=lambda i: aggregated[i]))
        return peak_idx, float(aggregated[peak_idx])

    elif method == "middle":
        n_frames = _count_frames(video_path)
        mid = n_frames // 2
        return mid, 0.0

    elif method == "attention_pool":
        logger.warning("attention_pool not implemented; falling back to middle")
        n_frames = _count_frames(video_path)
        mid = n_frames // 2
        return mid, 0.0

    else:
        raise ValueError(f"Unknown peak frame method: {method!r}")


def extract_frame_at_index(video_path: Path, frame_idx: int) -> Image.Image:
    """Extract a single frame at the given index as PIL Image.

    Args:
        video_path: Input video.
        frame_idx: 0-based frame index.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If video_path missing.
        ValueError: If frame cannot be read.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _aggregate_aus(au_intensities: dict[int, list[float]]) -> list[float]:
    """Sum AU intensities across all AUs per frame."""
    if not au_intensities:
        return []
    n = max(len(v) for v in au_intensities.values())
    agg = [0.0] * n
    for intensities in au_intensities.values():
        for i, val in enumerate(intensities):
            if i < n:
                agg[i] += val
    return agg


def _count_frames(video_path: Path) -> int:
    """Count total frames in video using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(1, n)
