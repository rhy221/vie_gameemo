"""Face cropping from webcam region.

Given a frame and the detected webcam bounding box, crop the streamer's face
region (with margin) and resize to target size for the face encoder.

If no webcam was detected (no-facecam mode), this module is bypassed and
the face encoder receives a zero placeholder.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from vie_gameemo.preprocess.webcam_detector import WebcamBBox

logger = logging.getLogger(__name__)

_face_detector = None


def _get_face_detector():
    """Return a shared FaceDetection instance (created once per process)."""
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp
        _face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0,
        )
    return _face_detector


def extract_streamer_face(
    frame: np.ndarray,
    webcam_bbox: WebcamBBox,
    target_size: tuple[int, int] = (224, 224),
    margin: float = 0.2,
    tight_crop: bool = True,
) -> np.ndarray:
    """Crop the streamer's face from a frame using the webcam region.

    Args:
        frame: Input frame as BGR ndarray (HxWx3).
        webcam_bbox: Detected webcam region (normalized coords).
        target_size: (width, height) for output.
        margin: Fractional margin to expand around the webcam region.
        tight_crop: If True, run MediaPipe inside webcam region for tighter crop.

    Returns:
        Cropped + resized face as BGR ndarray (target_size[1], target_size[0], 3).

    Raises:
        ValueError: If frame is empty.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided")

    h, w = frame.shape[:2]

    x1 = int((webcam_bbox.xmin - margin * webcam_bbox.width) * w)
    y1 = int((webcam_bbox.ymin - margin * webcam_bbox.height) * h)
    x2 = int((webcam_bbox.xmin + webcam_bbox.width * (1 + margin)) * w)
    y2 = int((webcam_bbox.ymin + webcam_bbox.height * (1 + margin)) * h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        logger.warning("Invalid crop region; returning black placeholder")
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    cropped = frame[y1:y2, x1:x2]

    if tight_crop:
        cropped = _tight_face_crop(cropped, fallback=cropped)

    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


def batch_extract_faces(
    frame_paths: list[Path],
    webcam_bbox: WebcamBBox | list[WebcamBBox | None],
    target_size: tuple[int, int] = (224, 224),
    margin: float = 0.2,
) -> np.ndarray:
    """Batch face extraction for an entire clip's frames.

    Args:
        frame_paths: Paths to extracted frames (JPG).
        webcam_bbox: Single bbox (same for all frames) or per-frame list.
        target_size: Output size (width, height).
        margin: Margin expansion.

    Returns:
        Stacked array of shape (N, H, W, 3) in BGR.
    """
    per_frame = isinstance(webcam_bbox, list)
    faces = []
    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Cannot read frame %s; using zeros", path)
            faces.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            continue
        bbox = webcam_bbox[i] if per_frame else webcam_bbox
        if bbox is None:
            faces.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            continue
        face = extract_streamer_face(frame, bbox, target_size, margin)
        faces.append(face)
    return np.stack(faces, axis=0)


def extract_webcam_region(
    frame: np.ndarray,
    webcam_bbox: WebcamBBox,
    target_size: tuple[int, int] = (224, 224),
    margin: float = 0.1,
    resize: bool = True,
) -> np.ndarray:
    """Crop the webcam region (wider than face crop, no tight crop).

    Args:
        frame: Input frame as BGR ndarray (HxWx3).
        webcam_bbox: Detected webcam region (normalized coords).
        target_size: (width, height) for output when resize=True.
        margin: Small margin to avoid cutting off edges.
        resize: If True (default), resize crop to target_size. Set False for
            pose branch — pose detection needs full native resolution, not 224×224.

    Returns:
        Cropped (and optionally resized) webcam region as BGR ndarray.
    """
    if frame is None or frame.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    h, w = frame.shape[:2]

    x1 = int((webcam_bbox.xmin - margin * webcam_bbox.width) * w)
    y1 = int((webcam_bbox.ymin - margin * webcam_bbox.height) * h)
    x2 = int((webcam_bbox.xmin + webcam_bbox.width * (1 + margin)) * w)
    y2 = int((webcam_bbox.ymin + webcam_bbox.height * (1 + margin)) * h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    cropped = frame[y1:y2, x1:x2]
    if resize:
        return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return cropped


def batch_extract_webcam_regions(
    frame_paths: list[Path],
    webcam_bbox: WebcamBBox | list[WebcamBBox | None],
    target_size: tuple[int, int] = (224, 224),
    margin: float = 0.1,
    resize: bool = True,
) -> list[np.ndarray]:
    """Batch webcam region extraction for context encoder.

    Args:
        frame_paths: Paths to extracted frames (JPG).
        webcam_bbox: Single bbox (same for all frames) or per-frame list.
        target_size: Output size (width, height) when resize=True.
        margin: Margin expansion.
        resize: If True (default), resize each crop to target_size. Pass False
            for the pose branch to preserve native resolution.

    Returns:
        List of BGR ndarrays, one per frame.
    """
    per_frame = isinstance(webcam_bbox, list)
    crops = []
    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Cannot read frame %s; using zeros", path)
            crops.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            continue
        bbox = webcam_bbox[i] if per_frame else webcam_bbox
        if bbox is None:
            crops.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            continue
        crops.append(extract_webcam_region(frame, bbox, target_size, margin, resize))
    return crops


def _tight_face_crop(region: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Run lightweight MediaPipe inside webcam region for a tighter face crop.

    Args:
        region: BGR crop of the webcam region.
        fallback: Returned as-is if MediaPipe finds no face in the region.

    Returns:
        Tighter face crop, or fallback if detection fails.
    """
    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        h, w = region.shape[:2]

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            detector = _get_face_detector()
            results = detector.process(rgb)
            if results.detections:
                bb = results.detections[0].location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width) * w))
                y2 = min(h, int((bb.ymin + bb.height) * h))
                if x2 > x1 and y2 > y1:
                    return region[y1:y2, x1:x2]
        else:
            from vie_gameemo.preprocess.webcam_detector import (
                _build_task_api_detector, _task_api_detect,
            )
            detector = _build_task_api_detector(0.5)
            bboxes = _task_api_detect(detector, rgb)
            if bboxes:
                bx, by, bw, bh = bboxes[0]
                x1 = max(0, int(bx * w))
                y1 = max(0, int(by * h))
                x2 = min(w, int((bx + bw) * w))
                y2 = min(h, int((by + bh) * h))
                if x2 > x1 and y2 > y1:
                    return region[y1:y2, x1:x2]
    except Exception as exc:
        logger.debug("Tight crop failed: %s", exc)
    return fallback
