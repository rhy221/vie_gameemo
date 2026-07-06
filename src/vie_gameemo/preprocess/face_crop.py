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

# Cached MediaPipe detector for _tight_face_crop, reused across frames so we
# don't spin up a new MediaPipe graph (and its internal thread pool) per call.
_tight_crop_detector = None
_tight_crop_use_task_api = False


def extract_streamer_face(
    frame: np.ndarray,
    webcam_bbox: WebcamBBox,
    target_size: tuple[int, int] = (224, 224),
    margin: float = 0.2,
    tight_crop: bool = True,
) -> tuple[np.ndarray, bool]:
    """Crop the streamer's face from a frame using the webcam region.

    Args:
        frame: Input frame as BGR ndarray (HxWx3).
        webcam_bbox: Detected webcam region (normalized coords).
        target_size: (width, height) for output.
        margin: Fractional margin to expand around the webcam region.
        tight_crop: If True, run MediaPipe inside webcam region for tighter crop.

    Returns:
        Tuple of (cropped + resized face as BGR ndarray, is_tight_face).
        `is_tight_face` is False when `tight_crop=True` but MediaPipe found no
        face in the webcam region (fell back to the wider, non-face-only
        crop), or when the crop region itself was invalid. True when
        `tight_crop=False` (no detection attempted, crop used as-is).

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
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8), False

    cropped = frame[y1:y2, x1:x2]

    is_tight_face = True
    if tight_crop:
        cropped, is_tight_face = _tight_face_crop(cropped, fallback=cropped)

    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized, is_tight_face


def batch_extract_faces(
    frame_paths: list[Path],
    webcam_bbox: WebcamBBox | list[WebcamBBox | None],
    target_size: tuple[int, int] = (224, 224),
    margin: float = 0.2,
) -> tuple[np.ndarray, list[bool]]:
    """Batch face extraction for an entire clip's frames.

    Args:
        frame_paths: Paths to extracted frames (JPG).
        webcam_bbox: Single bbox (same for all frames) or per-frame list.
        target_size: Output size (width, height).
        margin: Margin expansion.

    Returns:
        Tuple of:
            - Stacked array of shape (N, H, W, 3) in BGR.
            - List of N bools: True where the frame got a real tight face
              crop, False where it fell back to the wider webcam region (or
              had no bbox / unreadable frame) — lets callers avoid treating
              non-face frames as valid face data.
    """
    per_frame = isinstance(webcam_bbox, list)
    faces = []
    valid = []
    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Cannot read frame %s; using zeros", path)
            faces.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            valid.append(False)
            continue
        bbox = webcam_bbox[i] if per_frame else webcam_bbox
        if bbox is None:
            faces.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            valid.append(False)
            continue
        face, is_tight_face = extract_streamer_face(frame, bbox, target_size, margin)
        faces.append(face)
        valid.append(is_tight_face)
    return np.stack(faces, axis=0), valid


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


def _tight_face_crop(region: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    """Run lightweight MediaPipe inside webcam region for a tighter face crop.

    Args:
        region: BGR crop of the webcam region.
        fallback: Returned as-is if MediaPipe finds no face in the region.

    Returns:
        Tuple of (crop, is_tight_face). `is_tight_face` is False when
        MediaPipe found no face and `fallback` (the wider, un-refined webcam
        region — may contain background/hands/desk, not just a face) was
        returned instead. Callers use this to avoid silently mixing
        face-only crops and non-face fallback crops in the same sequence.
    """
    try:
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        h, w = region.shape[:2]
        detector, use_task_api = _get_tight_crop_detector()

        if use_task_api:
            from vie_gameemo.preprocess.webcam_detector import _task_api_detect
            bboxes = _task_api_detect(detector, rgb)
            if bboxes:
                bx, by, bw, bh = bboxes[0]
                x1 = max(0, int(bx * w))
                y1 = max(0, int(by * h))
                x2 = min(w, int((bx + bw) * w))
                y2 = min(h, int((by + bh) * h))
                if x2 > x1 and y2 > y1:
                    return region[y1:y2, x1:x2], True
        else:
            results = detector.process(rgb)
            if results.detections:
                bb = results.detections[0].location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width) * w))
                y2 = min(h, int((bb.ymin + bb.height) * h))
                if x2 > x1 and y2 > y1:
                    return region[y1:y2, x1:x2], True
    except Exception as exc:
        logger.debug("Tight crop failed: %s", exc)
    return fallback, False


def _get_tight_crop_detector():
    """Lazily create and cache the MediaPipe face detector used for tight crops.

    Returns:
        (detector, use_task_api) tuple. Cached at module level so repeated
        calls (one per frame) reuse the same MediaPipe graph instead of
        spinning up a new one each time.
    """
    global _tight_crop_detector, _tight_crop_use_task_api
    if _tight_crop_detector is not None:
        return _tight_crop_detector, _tight_crop_use_task_api

    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        _tight_crop_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0
        )
        _tight_crop_use_task_api = False
    else:
        from vie_gameemo.preprocess.webcam_detector import _build_task_api_detector
        _tight_crop_detector = _build_task_api_detector(0.5)
        _tight_crop_use_task_api = True

    return _tight_crop_detector, _tight_crop_use_task_api
