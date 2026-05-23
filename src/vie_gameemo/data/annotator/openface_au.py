"""OpenFace Action Unit extraction.

Wraps the OpenFace 2.x binary (FeatureExtraction) to extract per-frame
Action Unit (AU) intensities, which are used by:
- Peak frame detector (find frame with max emotional expression)
- Final annotation (Cved — visual expression description)

Requires OpenFace 2.x installed; see https://github.com/TadasBaltrusaitis/OpenFace
Configure binary path in config.annotation.openface.binary_path.
"""

import csv
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_AUS = [1, 2, 4, 5, 6, 7, 12, 15, 17, 20, 23, 25, 26, 45]


def extract_aus(
    video_path: Path,
    openface_binary: Path,
    output_dir: Path,
    target_aus: list[int] | None = None,
) -> dict[int, list[float]]:
    """Extract Action Unit intensities per frame using OpenFace.

    Args:
        video_path: Input video file.
        openface_binary: Path to OpenFace FeatureExtraction executable.
        output_dir: Where to save OpenFace CSV output.
        target_aus: AU codes to keep (e.g., [1, 2, 4, 6, 12]). None = default set.

    Returns:
        Dict mapping AU code → list of per-frame intensities.

    Raises:
        FileNotFoundError: If video or binary missing.
        RuntimeError: If OpenFace fails.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not openface_binary.exists():
        raise FileNotFoundError(
            f"OpenFace binary not found: {openface_binary}. "
            "See https://github.com/TadasBaltrusaitis/OpenFace for installation."
        )

    target_aus = target_aus or _DEFAULT_TARGET_AUS
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{video_path.stem}.csv"
    if csv_path.exists():
        logger.debug("AU CSV already exists (skip): %s", csv_path)
        return _parse_au_csv(csv_path, target_aus)

    cmd = [
        str(openface_binary),
        "-f", str(video_path),
        "-out_dir", str(output_dir),
        "-aus",
    ]
    # OpenFace resolves its model directory ("model/") relative to the
    # binary's location (i.e. build/bin/model after a standard CMake build).
    # Run from that directory so model lookup is unambiguous.
    run_cwd = openface_binary.parent
    model_dir = run_cwd / "model"
    if not model_dir.exists():
        raise RuntimeError(
            f"OpenFace model directory missing: {model_dir}. "
            "Did download_models.sh complete? "
            f"Listing of {run_cwd}: {sorted(p.name for p in run_cwd.iterdir()) if run_cwd.exists() else '<missing>'}"
        )

    logger.info("Running OpenFace on %s (cwd=%s)", video_path.name, run_cwd)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_cwd))

    # OpenFace sometimes exits non-zero but still writes a valid CSV; only
    # treat as failure if the CSV is also missing. Surface stdout+stderr
    # because OpenFace writes most diagnostics to stdout.
    if not csv_path.exists():
        diag = ((result.stderr or "") + "\n" + (result.stdout or "")).strip() or "(no output)"
        raise RuntimeError(
            f"OpenFace failed on {video_path} (rc={result.returncode}, cwd={run_cwd}): "
            f"{diag[:800]}"
        )
    if result.returncode != 0:
        logger.warning(
            "OpenFace returned rc=%d on %s but CSV exists — continuing. stderr=%r",
            result.returncode, video_path.name, (result.stderr or "")[:200],
        )

    return _parse_au_csv(csv_path, target_aus)


def aggregate_au_intensity(au_intensities: dict[int, list[float]]) -> list[float]:
    """Sum AU intensities per frame across all target AUs.

    Args:
        au_intensities: AU code → per-frame intensity list.

    Returns:
        List of per-frame aggregated scores (higher = more expression).
    """
    if not au_intensities:
        return []

    n_frames = max(len(v) for v in au_intensities.values())
    aggregated = [0.0] * n_frames
    for intensities in au_intensities.values():
        for i, val in enumerate(intensities):
            if i < n_frames:
                aggregated[i] += val
    return aggregated


def _parse_au_csv(
    csv_path: Path,
    target_aus: list[int],
) -> dict[int, list[float]]:
    """Parse OpenFace AU intensity CSV output.

    Args:
        csv_path: Path to OpenFace output CSV.
        target_aus: AU codes to extract.

    Returns:
        Dict of AU code → list of intensities.
    """
    result: dict[int, list[float]] = {au: [] for au in target_aus}

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for au in target_aus:
                col_key = f" AU{au:02d}_r"
                alt_key = f"AU{au:02d}_r"
                val_str = row.get(col_key, row.get(alt_key, "0")).strip()
                try:
                    result[au].append(float(val_str))
                except ValueError:
                    result[au].append(0.0)

    return result
