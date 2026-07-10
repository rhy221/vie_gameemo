"""Raw-image augmentation for face/context crops (BGR uint8 ndarrays).

Operates on the same BGR ndarray format used throughout `preprocess/face_crop.py`
and the context encoder's webcam-region crops — applied BEFORE encoding, unlike
`training.losses.embedding_augment` which perturbs the already-encoded feature.
"""

import random

import cv2
import numpy as np


def random_color_jitter(
    img: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05,
) -> np.ndarray:
    """Randomly perturb brightness/contrast/saturation/hue of a BGR image.

    Args:
        img: BGR uint8 ndarray (H, W, 3).
        brightness, contrast, saturation: Max relative jitter fraction (0-1),
            each independently sampled from U(-x, x). 0 disables that jitter.
        hue: Max hue shift as a fraction of OpenCV's 180-degree hue range.

    Returns:
        Jittered BGR uint8 ndarray, same shape as input (new array).
    """
    out = img.astype(np.float32)

    if brightness > 0:
        out = out * (1.0 + random.uniform(-brightness, brightness))

    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        out = (out - out.mean()) * factor + out.mean()

    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if saturation > 0:
            hsv[..., 1] = hsv[..., 1] * (1.0 + random.uniform(-saturation, saturation))
        if hue > 0:
            shift = random.uniform(-hue, hue) * 180.0
            hsv[..., 0] = (hsv[..., 0] + shift) % 180.0
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def random_crop_resize(
    img: np.ndarray,
    target_size: tuple[int, int] | None = None,
    scale_range: tuple[float, float] = (0.85, 1.0),
) -> np.ndarray:
    """Randomly crop a sub-region (by area fraction) and resize back.

    Args:
        img: BGR uint8 ndarray (H, W, 3).
        target_size: (width, height) to resize the crop to. Defaults to the
            input's own (W, H) so output shape always matches input shape.
        scale_range: Min/max fraction of the original AREA the crop keeps.

    Returns:
        Cropped + resized BGR uint8 ndarray.
    """
    h, w = img.shape[:2]
    target_size = target_size or (w, h)

    scale = random.uniform(*scale_range)
    crop_h = max(1, int(round(h * scale ** 0.5)))
    crop_w = max(1, int(round(w * scale ** 0.5)))
    y0 = random.randint(0, max(0, h - crop_h))
    x0 = random.randint(0, max(0, w - crop_w))

    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)


def augment_image(
    img: np.ndarray,
    color_jitter: dict | None = None,
    crop_scale_range: tuple[float, float] | None = None,
    horizontal_flip_p: float = 0.0,
) -> np.ndarray:
    """Apply the configured combination of image augmentations to one frame.

    Args:
        img: BGR uint8 ndarray (H, W, 3).
        color_jitter: Kwargs forwarded to `random_color_jitter`, or None/empty to skip.
        crop_scale_range: `scale_range` for `random_crop_resize`, or None to skip.
        horizontal_flip_p: Probability of a horizontal flip.

    Returns:
        Augmented BGR uint8 ndarray, same shape as input.
    """
    out = img
    if crop_scale_range is not None:
        out = random_crop_resize(out, target_size=(img.shape[1], img.shape[0]), scale_range=crop_scale_range)
    if color_jitter:
        out = random_color_jitter(out, **color_jitter)
    if horizontal_flip_p > 0 and random.random() < horizontal_flip_p:
        out = np.ascontiguousarray(out[:, ::-1])
    return out
