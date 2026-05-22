"""Demuxing: extract audio and video frames from clip files using ffmpeg + opencv.

Stage 1 of the pipeline. Outputs:
    - data/processed/audios/{clip_id}.wav  (16kHz mono)
    - data/processed/frames/{clip_id}/frame_0000.jpg, ...
"""

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Extract audio from video using ffmpeg.

    Args:
        video_path: Input video file.
        output_path: Output wav path.
        sample_rate: Target sample rate (default 16kHz for speech).
        channels: Mono (1) or stereo (2).

    Returns:
        Path to extracted audio.

    Raises:
        FileNotFoundError: If video_path missing.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path.exists():
        logger.debug("Audio already extracted (skip): %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_path),
    ]
    logger.debug("ffmpeg audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info("Extracted audio: %s", output_path)
    return output_path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    quality: int = 90,
) -> list[Path]:
    """Extract frames from video at given fps using OpenCV.

    Args:
        video_path: Input video.
        output_dir: Output directory (created if missing).
        fps: Frames per second to extract.
        quality: JPG quality (1-100).

    Returns:
        List of paths to extracted frames, in temporal order.

    Raises:
        FileNotFoundError: If video_path missing.
        RuntimeError: If video cannot be opened.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        logger.debug("Frames already extracted (skip): %s (%d frames)", output_dir, len(existing))
        return existing

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(video_fps / fps)))
    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    saved: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                out_path = output_dir / f"frame_{saved_idx:04d}.jpg"
                cv2.imwrite(str(out_path), frame, jpg_params)
                saved.append(out_path)
                saved_idx += 1
            frame_idx += 1
    finally:
        cap.release()

    logger.info("Extracted %d frames → %s", len(saved), output_dir)
    return saved


def separate_vocals(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
) -> Path:
    """Separate vocals from background music using Demucs.

    Only call if EDA shows music is too loud relative to speech.

    Args:
        audio_path: Input audio wav.
        output_dir: Output directory for separated stems.
        model: Demucs model name.

    Returns:
        Path to vocals-only wav.

    Raises:
        FileNotFoundError: If audio_path missing.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "demucs",
        "--model", model,
        "--out", str(output_dir),
        "--two-stems", "vocals",
        str(audio_path),
    ]
    logger.info("Running Demucs: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)

    vocals_path = output_dir / model / audio_path.stem / "vocals.wav"
    if not vocals_path.exists():
        raise RuntimeError(f"Demucs did not produce vocals at: {vocals_path}")
    return vocals_path
