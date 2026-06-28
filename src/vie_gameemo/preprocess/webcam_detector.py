"""Webcam region detection for streamer face localization.

In livestream gaming, the streamer's webcam occupies only 10-15% of the
frame (typically a corner). To distinguish the streamer's face from
in-game NPC faces (cutscenes, etc.), we use spatial stability: the webcam
appears at a stable position across all frames, while NPC faces are
sporadic and central.

Supports 3 detection backends (configured via config.yaml):
    - mediapipe: BlazeFace (fast, lightweight, may miss small webcams)
    - owlv2:    OWLv2 zero-shot detector with "facecam overlay" prompt (accurate, slower)
    - yolo:     YOLOv11 (fast + accurate, requires trained weights)

Approach (Section 5.3 of spec):
    1. Sample N frames evenly across the clip
    2. Run detection backend on each frame
    3. Pick the detection with highest confidence as the webcam bbox
    4. Optionally cluster + validate stability across frames
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
    """Detect streamer webcam region. Supports mediapipe, owlv2, and yolo backends."""

    def __init__(
        self,
        backend: str = "mediapipe",
        min_detection_confidence: float = 0.4,
        sample_n_frames: int = 30,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 5,
        stability_threshold: float = 0.5,
        edge_bias: float = 0.3,
        owlv2_model: str = "google/owlv2-base-patch16-finetuned",
        owlv2_prompt: str = "facecam overlay",
        yolo_model: str = "yolo11n.pt",
        yolo_classes: list[int] | None = None,
    ) -> None:
        self.backend = backend
        self.min_detection_confidence = min_detection_confidence
        self.sample_n_frames = sample_n_frames
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.stability_threshold = stability_threshold
        self.edge_bias = edge_bias
        self.owlv2_model = owlv2_model
        self.owlv2_prompt = owlv2_prompt
        self.yolo_model = yolo_model
        self.yolo_classes = yolo_classes if yolo_classes is not None else [0]
        self._detector = None

    def _init_detector(self) -> None:
        """Lazy-initialize the detection backend."""
        if self._detector is not None:
            return
        if self.backend == "owlv2":
            self._detector = _OWLv2Backend(self.owlv2_model, self.owlv2_prompt, self.min_detection_confidence)
        elif self.backend == "yolo":
            self._detector = _YOLOBackend(self.yolo_model, self.min_detection_confidence, self.yolo_classes)
        else:
            self._detector = _MediaPipeBackend(self.min_detection_confidence)

    def detect_webcam_region(self, clip_path: Path) -> WebcamBBox | None:
        """Detect webcam region in a clip.

        Samples frames, runs backend, picks highest-confidence detection,
        then validates via clustering for stability.

        Returns:
            WebcamBBox if a stable webcam region found, None otherwise.
        """
        clip_path = Path(clip_path)
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        self._init_detector()
        frames = self._sample_frames(clip_path)
        if not frames:
            logger.warning("No frames sampled from %s", clip_path)
            return None

        detections = self._detect_all_frames(frames)
        if not detections:
            logger.info("No detections in %s (backend=%s)", clip_path.name, self.backend)
            return None

        return self._cluster_detections(detections, n_sampled=len(frames))

    def detect_per_frame(
        self, frames: list[np.ndarray],
    ) -> list[WebcamBBox | None]:
        """Detect face bbox per frame with cluster-based fallback."""
        self._init_detector()
        if not frames:
            return []

        all_detections = self._detect_all_frames(frames)
        stable_bbox = self._cluster_detections(all_detections, n_sampled=len(frames))

        per_frame: list[WebcamBBox | None] = []
        for frame in frames:
            bboxes = self._detector.detect_frame(frame)
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
        """Convenience: sample frames from clip then detect per-frame."""
        if not Path(clip_path).exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")
        old_n = self.sample_n_frames
        if n_frames is not None:
            self.sample_n_frames = n_frames
        frames = self._sample_frames(clip_path)
        self.sample_n_frames = old_n
        return self.detect_per_frame(frames)

    def _detect_all_frames(
        self, frames: list[np.ndarray],
    ) -> list[tuple[float, float, float, float]]:
        """Run backend on all frames, return normalized bboxes."""
        detections: list[tuple[float, float, float, float]] = []
        for frame in frames:
            bboxes = self._detector.detect_frame(frame)
            detections.extend(bboxes)
        return detections

    def _pick_best_bbox(
        self,
        bboxes: list[tuple[float, float, float, float]],
        stable_bbox: WebcamBBox | None,
    ) -> WebcamBBox:
        """Pick the bbox closest to the stable webcam region."""
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
        """Sample N frames evenly across the clip duration."""
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

    def _cluster_detections(
        self,
        detections: list[tuple[float, float, float, float]],
        n_sampled: int,
    ) -> WebcamBBox | None:
        """Cluster detections → estimate webcam region.

        Uses only small detections (< 25% frame area) for clustering.
        After finding the stable cluster, expands bbox to approximate
        the full webcam overlay region.
        """
        from sklearn.cluster import DBSCAN

        if not detections:
            return None

        max_area_for_clustering = 0.25
        cluster_candidates = [
            (x, y, w, h) for x, y, w, h in detections
            if w * h < max_area_for_clustering
        ]
        if not cluster_candidates:
            cluster_candidates = list(detections)

        centers = np.array([(x + w / 2, y + h / 2) for x, y, w, h in cluster_candidates])
        labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(centers)

        best_cluster: int | None = None
        best_score = -1.0

        for label in set(labels) - {-1}:
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
        cluster_dets = [cluster_candidates[i] for i, m in enumerate(mask) if m]

        face_xmin = float(np.median([d[0] for d in cluster_dets]))
        face_ymin = float(np.median([d[1] for d in cluster_dets]))
        face_w = float(np.median([d[2] for d in cluster_dets]))
        face_h = float(np.median([d[3] for d in cluster_dets]))

        expand_x, expand_y = 1.0, 1.5
        xmin = max(0.0, face_xmin - face_w * expand_x * 0.5)
        ymin = max(0.0, face_ymin - face_h * expand_y * 0.3)
        width = min(1.0 - xmin, face_w * (1.0 + expand_x))
        height = min(1.0 - ymin, face_h * (1.0 + expand_y))

        cx, cy = xmin + width / 2, ymin + height / 2
        edge_distance = float(min(cx, cy, 1.0 - cx, 1.0 - cy))
        stability_score = float(mask.sum()) / n_sampled

        return WebcamBBox(
            xmin=xmin, ymin=ymin, width=width, height=height,
            stability_score=stability_score, edge_distance=edge_distance,
        )


# ---------------------------------------------------------------------------
# Backend: MediaPipe
# ---------------------------------------------------------------------------

class _MediaPipeBackend:
    """BlazeFace face detection via MediaPipe."""

    def __init__(self, min_confidence: float = 0.4) -> None:
        self.min_confidence = min_confidence
        self._detector = None
        self._use_task_api = False

    def _init(self) -> None:
        if self._detector is not None:
            return
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._detector = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=self.min_confidence,
                    model_selection=1,
                )
            else:
                self._detector = _build_task_api_detector(self.min_confidence)
                self._use_task_api = True
        except ImportError as e:
            raise ImportError("mediapipe not installed. Run: pip install mediapipe") from e

    def detect_frame(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Detect faces in a single BGR frame. Returns normalized bboxes."""
        self._init()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = self._detect_rgb(rgb)
        if not bboxes:
            bboxes = self._detect_in_corners(rgb)
        return bboxes

    def _detect_rgb(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        if self._use_task_api:
            return _task_api_detect(self._detector, rgb)
        results = self._detector.process(rgb)
        if not results.detections:
            return []
        return [
            (max(0.0, bb.xmin), max(0.0, bb.ymin), min(1.0, bb.width), min(1.0, bb.height))
            for det in results.detections
            for bb in [det.location_data.relative_bounding_box]
        ]

    def _detect_in_corners(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Crop each corner quadrant, upscale, and retry detection."""
        h, w = rgb.shape[:2]
        crop_ratio = 0.35
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        corners = [
            (0, 0), (0, w - cw), (h - ch, 0), (h - ch, w - cw),
        ]
        all_bboxes: list[tuple[float, float, float, float]] = []
        for y0, x0 in corners:
            crop = rgb[y0:y0 + ch, x0:x0 + cw]
            scale = max(1.0, 320.0 / min(crop.shape[:2]))
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            for bx, by, bw, bh in self._detect_rgb(crop):
                all_bboxes.append(((x0 + bx * cw) / w, (y0 + by * ch) / h, (bw * cw) / w, (bh * ch) / h))
        return all_bboxes


# ---------------------------------------------------------------------------
# Backend: OWLv2 (zero-shot object detection)
# ---------------------------------------------------------------------------

class _OWLv2Backend:
    """OWLv2 zero-shot detector — detects webcam overlay by text prompt."""

    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-finetuned",
        prompt: str = "facecam overlay",
        min_confidence: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.min_confidence = min_confidence
        self._model = None
        self._processor = None
        self._device = None

    def _init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as e:
            raise ImportError(
                "transformers not installed or OWLv2 not available. "
                "Run: pip install transformers>=4.36.0"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading OWLv2: %s (device=%s)", self.model_name, self._device)
        try:
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self._device)
        except (ValueError, OSError):
            # Fallback: safetensors available on PR branch for older torch
            logger.info("Retrying OWLv2 with safetensors revision...")
            self._processor = Owlv2Processor.from_pretrained(self.model_name, revision="refs/pr/5")
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name, revision="refs/pr/5").to(self._device)
        self._model.eval()
        logger.info("OWLv2 loaded (prompt=%r)", self.prompt)

    def detect_frame(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Detect webcam overlay in a single BGR frame."""
        self._init()
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        h, w = frame.shape[:2]

        inputs = self._processor(text=[[self.prompt]], images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=self.min_confidence, target_sizes=target_sizes,
        )[0]

        bboxes = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        for box, score in zip(boxes, scores):
            y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
            bboxes.append((
                float(x1 / w), float(y1 / h),
                float((x2 - x1) / w), float((y2 - y1) / h),
            ))

        if bboxes:
            logger.debug("OWLv2: %d detections (best score=%.3f)", len(bboxes), scores.max())

        return bboxes


# ---------------------------------------------------------------------------
# Backend: YOLOv11 (ultralytics)
# ---------------------------------------------------------------------------

class _YOLOBackend:
    """YOLOv11 object detection via ultralytics."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        min_confidence: float = 0.4,
        classes: list[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.classes = classes if classes is not None else [0]
        self._model = None

    def _init(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e

        logger.info("Loading YOLO: %s (classes=%s)", self.model_path, self.classes)
        self._model = YOLO(self.model_path)
        logger.info("YOLO loaded")

    def detect_frame(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Detect persons/faces in a single BGR frame."""
        self._init()
        h, w = frame.shape[:2]

        results = self._model.predict(
            frame,
            conf=self.min_confidence,
            classes=self.classes,
            verbose=False,
        )

        bboxes = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bboxes.append((
                    float(x1 / w), float(y1 / h),
                    float((x2 - x1) / w), float((y2 - y1) / h),
                ))

        return bboxes


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
    """Download the blaze_face_full_range.tflite model if not cached."""
    import urllib.request

    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "blaze_face_full_range.tflite"
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_full_range/float16/latest/"
            "blaze_face_full_range.tflite"
        )
        logger.info("Downloading MediaPipe face model (full-range) → %s", model_path)
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
