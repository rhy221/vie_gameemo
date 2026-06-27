"""Webcam region detection for streamer face localization.

In livestream gaming, the streamer's webcam occupies only 10-15% of the
frame (typically a corner). To distinguish the streamer's face from
in-game NPC faces (cutscenes, etc.), we use spatial stability: the webcam
appears at a stable position across all frames, while NPC faces are
sporadic and central.

Approach (Section 5.3 of spec):
    1. Sample N frames evenly across the clip
    2. Run MediaPipe FaceDetection on each
    3. Cluster face bounding boxes by position using DBSCAN
    4. Pick the largest cluster (most stable position) as the webcam region
    5. Optionally bias toward edge positions (webcam usually in a corner)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WebcamBBox:
    """Detected webcam region in normalized coords (0.0–1.0)."""
    xmin: float
    ymin: float
    width: float
    height: float
    stability_score: float
    edge_distance: float


class WebcamDetector:
    """Detect streamer webcam region using stable face clustering."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        sample_n_frames: int = 30,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 5,
        stability_threshold: float = 0.5,
        edge_bias: float = 0.3,
    ) -> None:
        """Initialize webcam detector.

        Args:
            min_detection_confidence: MediaPipe face detection threshold.
            sample_n_frames: Number of frames to sample across clip.
            dbscan_eps: DBSCAN epsilon (cluster radius in normalized coords).
            dbscan_min_samples: Min cluster size for DBSCAN.
            stability_threshold: cluster_size / sample_n must exceed this.
            edge_bias: Webcam typically within this normalized distance from edge.
        """
        self.min_detection_confidence = min_detection_confidence
        self.sample_n_frames = sample_n_frames
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.stability_threshold = stability_threshold
        self.edge_bias = edge_bias
        self._detector = None
        self._use_task_api = False

    def _init_detector(self) -> None:
        """Lazy-initialize the MediaPipe face detection model."""
        if self._detector is not None:
            return
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._detector = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=self.min_detection_confidence,
                    model_selection=0,
                )
                self._use_task_api = False
            else:
                self._detector = _build_task_api_detector(self.min_detection_confidence)
                self._use_task_api = True
        except ImportError as e:
            raise ImportError("mediapipe not installed. Run: pip install mediapipe") from e

    def detect_webcam_region(self, clip_path: Path) -> WebcamBBox | None:
        """Detect webcam region in a clip (stable cluster across sampled frames).

        Args:
            clip_path: Path to video file.

        Returns:
            WebcamBBox if a stable face cluster found, None otherwise.

        Raises:
            FileNotFoundError: If clip_path missing.
        """
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        self._init_detector()
        frames = self._sample_frames(clip_path)
        if not frames:
            logger.warning("No frames sampled from %s", clip_path)
            return None

        detections = self._detect_faces_in_frames(frames)
        if not detections:
            logger.info("No faces detected in %s", clip_path)
            return None

        return self._cluster_detections(detections, n_sampled=len(frames))

    def detect_per_frame(
        self, frames: list[np.ndarray],
    ) -> list[WebcamBBox | None]:
        """Detect face bbox per frame with cluster-based fallback.

        For each frame, returns the best face bbox. When a frame has no
        detection, falls back to the stable cluster bbox (if available).
        Handles webcam zoom/resize mid-clip.

        Args:
            frames: List of BGR frames.

        Returns:
            List of WebcamBBox (one per frame), None where no face found
            even with fallback.
        """
        self._init_detector()
        if not frames:
            return []

        all_detections = self._detect_faces_in_frames(frames)
        stable_bbox = self._cluster_detections(all_detections, n_sampled=len(frames))

        per_frame: list[WebcamBBox | None] = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = self._detect_single_frame(rgb)
            if bboxes:
                best = self._pick_best_bbox(bboxes, stable_bbox)
                per_frame.append(best)
            elif stable_bbox is not None:
                per_frame.append(stable_bbox)
            else:
                per_frame.append(None)

        return per_frame

    def detect_per_frame_from_clip(
        self, clip_path: Path, n_frames: int | None = None,
    ) -> list[WebcamBBox | None]:
        """Convenience: sample frames from clip then detect per-frame.

        Args:
            clip_path: Path to video file.
            n_frames: Number of frames to sample. Defaults to sample_n_frames.

        Returns:
            List of per-frame WebcamBBox.
        """
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        old_n = self.sample_n_frames
        if n_frames is not None:
            self.sample_n_frames = n_frames
        frames = self._sample_frames(clip_path)
        self.sample_n_frames = old_n

        return self.detect_per_frame(frames)

    def _pick_best_bbox(
        self,
        bboxes: list[tuple[float, float, float, float]],
        stable_bbox: WebcamBBox | None,
    ) -> WebcamBBox:
        """Pick the bbox closest to the stable webcam region.

        When zoom happens, the face bbox changes size but stays near
        the same area. Picks the bbox closest to the stable center,
        or the largest face if no stable reference.
        """
        candidates = []
        for x, y, w, h in bboxes:
            cx, cy = x + w / 2, y + h / 2
            edge_dist = float(min(cx, cy, 1.0 - cx, 1.0 - cy))
            candidates.append((x, y, w, h, cx, cy, edge_dist, w * h))

        if stable_bbox is not None:
            ref_cx = stable_bbox.xmin + stable_bbox.width / 2
            ref_cy = stable_bbox.ymin + stable_bbox.height / 2
            candidates.sort(key=lambda c: (c[4] - ref_cx) ** 2 + (c[5] - ref_cy) ** 2)
        else:
            candidates.sort(key=lambda c: c[7], reverse=True)

        best = candidates[0]
        return WebcamBBox(
            xmin=best[0], ymin=best[1], width=best[2], height=best[3],
            stability_score=1.0, edge_distance=best[6],
        )

    def _sample_frames(self, clip_path: Path) -> list[np.ndarray]:
        """Sample N frames evenly across the clip duration.

        Args:
            clip_path: Path to video file.

        Returns:
            List of BGR frames as numpy arrays.
        """
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        n = min(self.sample_n_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def _detect_faces_in_frames(
        self, frames: list[np.ndarray]
    ) -> list[tuple[float, float, float, float]]:
        """Run MediaPipe on each frame; return list of (xmin, ymin, w, h) in [0,1].

        Args:
            frames: List of BGR frames.

        Returns:
            List of normalized bounding boxes.
        """
        detections: list[tuple[float, float, float, float]] = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = self._detect_single_frame(rgb)
            detections.extend(bboxes)
        return detections

    def _detect_single_frame(
        self, rgb: np.ndarray
    ) -> list[tuple[float, float, float, float]]:
        """Detect faces in a single RGB frame, compatible with both APIs."""
        if self._use_task_api:
            return _task_api_detect(self._detector, rgb)
        results = self._detector.process(rgb)
        if not results.detections:
            return []
        bboxes = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            bboxes.append((
                max(0.0, bb.xmin),
                max(0.0, bb.ymin),
                min(1.0, bb.width),
                min(1.0, bb.height),
            ))
        return bboxes

    def _cluster_detections(
        self,
        detections: list[tuple[float, float, float, float]],
        n_sampled: int,
    ) -> WebcamBBox | None:
        """Cluster detection positions; return average bbox of most stable cluster.

        Args:
            detections: Normalized bounding boxes (xmin, ymin, w, h).
            n_sampled: Total frames sampled (for stability score).

        Returns:
            WebcamBBox of winning cluster, or None if threshold not met.
        """
        from sklearn.cluster import DBSCAN

        if not detections:
            return None

        centers = np.array([(x + w / 2, y + h / 2) for x, y, w, h in detections])
        labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(centers)

        best_cluster: int | None = None
        best_score = -1.0

        unique_labels = set(labels) - {-1}
        for label in unique_labels:
            mask = labels == label
            cluster_size = int(mask.sum())
            stability = cluster_size / n_sampled
            if stability < self.stability_threshold:
                continue

            cluster_centers = centers[mask]
            cx, cy = cluster_centers.mean(axis=0)
            edge_dist = min(cx, cy, 1.0 - cx, 1.0 - cy)
            score = stability - edge_dist

            if score > best_score:
                best_score = score
                best_cluster = label

        if best_cluster is None:
            return None

        mask = labels == best_cluster
        cluster_dets = [detections[i] for i, m in enumerate(mask) if m]
        xmin = float(np.mean([d[0] for d in cluster_dets]))
        ymin = float(np.mean([d[1] for d in cluster_dets]))
        width = float(np.mean([d[2] for d in cluster_dets]))
        height = float(np.mean([d[3] for d in cluster_dets]))

        cx, cy = xmin + width / 2, ymin + height / 2
        edge_distance = float(min(cx, cy, 1.0 - cx, 1.0 - cy))
        stability_score = float(mask.sum()) / n_sampled

        return WebcamBBox(
            xmin=xmin,
            ymin=ymin,
            width=width,
            height=height,
            stability_score=stability_score,
            edge_distance=edge_distance,
        )


# ---------------------------------------------------------------------------
# MediaPipe Task API helpers (mediapipe >= 0.10.18)
# ---------------------------------------------------------------------------

def _build_task_api_detector(min_confidence: float):
    """Build a face detector using the new MediaPipe Task API."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    model_path = _ensure_task_model()
    options = vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=min_confidence,
    )
    return vision.FaceDetector.create_from_options(options)


def _ensure_task_model() -> Path:
    """Download the blaze_face_short_range.tflite model if not cached."""
    import urllib.request

    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "blaze_face_short_range.tflite"
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/latest/"
            "blaze_face_short_range.tflite"
        )
        logger.info("Downloading MediaPipe face model → %s", model_path)
        urllib.request.urlretrieve(url, model_path)
    return model_path


def _task_api_detect(
    detector, rgb: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """Run the Task API detector on an RGB frame."""
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    bboxes = []
    h, w = rgb.shape[:2]
    for det in result.detections:
        bb = det.bounding_box
        bboxes.append((
            max(0.0, bb.origin_x / w),
            max(0.0, bb.origin_y / h),
            min(1.0, bb.width / w),
            min(1.0, bb.height / h),
        ))
    return bboxes
