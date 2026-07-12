"""Shared per-clip feature-extraction logic.

Used by `scripts/extract_features.py` (offline, cached-mode extraction) AND
by the "online" raw-augmentation training path (`training/perception.py`),
so both go through the exact same crop-resolution / bbox-fallback / encoder
call logic instead of two divergent copies.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch


def index_videos(videos_dir: Path) -> dict[str, Path]:
    """Map clip_id (stem) → source video path, searching recursively.

    Videos may be nested under split/label subfolders
    (raw_videos/<split>/<label>/<clip_id>.mp4).
    """
    index: dict[str, Path] = {}
    if not videos_dir.exists():
        return index
    for ext in ("*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov"):
        for vp in videos_dir.glob(f"**/{ext}"):
            index.setdefault(vp.stem, vp)
    return index


def build_augment_fns(cfg) -> tuple[Callable[[np.ndarray], np.ndarray] | None, Callable | None]:
    """Build the image/audio augmentation callables from `cfg.augment.raw`.

    Returns:
        (image_aug_fn, audio_aug_fn): image_aug_fn takes/returns a BGR uint8
        crop; audio_aug_fn takes (wav, sr) and returns an augmented wav. Both
        None if `augment.raw` (or the respective `image`/`audio` sub-block)
        is unset — caller should treat that as "no augmentation" for that modality.
    """
    raw_cfg = getattr(getattr(cfg, "augment", None), "raw", None)
    if raw_cfg is None:
        return None, None

    img_cfg = getattr(raw_cfg, "image", None)
    color_jitter = None
    crop_scale_range = None
    flip_p = 0.0
    if img_cfg is not None:
        cj = getattr(img_cfg, "color_jitter", None)
        if cj is not None:
            color_jitter = {
                "brightness": getattr(cj, "brightness", 0.2),
                "contrast": getattr(cj, "contrast", 0.2),
                "saturation": getattr(cj, "saturation", 0.2),
                "hue": getattr(cj, "hue", 0.05),
            }
        crop_scale = getattr(img_cfg, "crop_scale_range", None)
        crop_scale_range = tuple(crop_scale) if crop_scale is not None else None
        flip_p = getattr(img_cfg, "horizontal_flip_p", 0.0)

    def image_aug_fn(img: np.ndarray) -> np.ndarray:
        from vie_gameemo.augment.image_augment import augment_image
        return augment_image(
            img, color_jitter=color_jitter, crop_scale_range=crop_scale_range,
            horizontal_flip_p=flip_p,
        )

    aud_cfg = getattr(raw_cfg, "audio", None)
    pitch_range = time_range = noise_range = spec_kwargs = None
    if aud_cfg is not None:
        pr = getattr(aud_cfg, "pitch_shift_semitones", None)
        pitch_range = tuple(pr) if pr is not None else None
        tr = getattr(aud_cfg, "time_stretch_rate", None)
        time_range = tuple(tr) if tr is not None else None
        nr = getattr(aud_cfg, "noise_std", None)
        noise_range = tuple(nr) if nr is not None else None
        sa = getattr(aud_cfg, "spec_augment", None)
        if sa is not None:
            spec_kwargs = {
                "time_mask_p": getattr(sa, "time_mask_p", 0.2),
                "freq_mask_p": getattr(sa, "freq_mask_p", 0.2),
            }

    def audio_aug_fn(wav: np.ndarray, sr: int) -> np.ndarray:
        from vie_gameemo.augment.audio_augment import augment_waveform
        return augment_waveform(
            wav, sr, pitch_shift_semitones=pitch_range, time_stretch_rate=time_range,
            noise_std=noise_range, spec_augment=spec_kwargs,
        )

    return (image_aug_fn if img_cfg is not None else None), (audio_aug_fn if aud_cfg is not None else None)


def extract_clip_features(
    cfg,
    encoders: dict,
    clip_id: str,
    ann,
    resolved_bbox,
    has_face: bool,
    frame_paths: list[Path],
    strategy: str,
    video_index: dict[str, Path],
    logger,
    image_aug: Callable[[np.ndarray], np.ndarray] | None = None,
    audio_aug: Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> dict[str, torch.Tensor]:
    """Build the features dict for one clip (all requested modalities).

    Args:
        image_aug: If given, applied to every face/context crop right after
            cropping, before encoding.
        audio_aug: If given, applied to the loaded waveform before encoding
            (loads via librosa itself instead of handing the encoder a Path,
            since the encoder needs to see the augmented samples).

    Returns:
        Features dict with whatever modality keys are present in `encoders`,
        plus `has_face`.
    """
    features: dict[str, torch.Tensor] = {}

    if "audio" in encoders:
        audio_path = Path(cfg.paths.audios) / f"{clip_id}.wav"
        source = audio_path if audio_path.exists() else video_index.get(clip_id)
        if source is not None:
            if audio_aug is not None:
                import librosa
                sr = encoders["audio"].sample_rate
                wav, _ = librosa.load(str(source), sr=sr, mono=True)
                wav = audio_aug(wav, sr)
                features["audio"] = encoders["audio"].encode(wav).squeeze(0)
            else:
                features["audio"] = encoders["audio"].encode(source).squeeze(0)
        else:
            logger.warning("No audio source (wav or video) for: %s", clip_id)
            features["audio"] = torch.zeros(
                cfg.audio_encoder.target_tokens, encoders["audio"].d_out
            )

    if "face" in encoders:
        if strategy == "full_frame":
            from vie_gameemo.preprocess.face_crop import _tight_face_crop
            crops, valid_mask = [], []
            for fp in frame_paths:
                frame = cv2.imread(str(fp))
                if frame is not None:
                    cropped, is_tight_face = _tight_face_crop(frame, fallback=frame)
                    if image_aug is not None:
                        cropped = image_aug(cropped)
                    crops.append(cropped)
                    valid_mask.append(is_tight_face)
            feat, _ = encoders["face"].encode(crops if crops else None, valid_mask=valid_mask)
            features["face"] = feat.squeeze(0)
        elif has_face and resolved_bbox is not None and frame_paths:
            from vie_gameemo.preprocess.face_crop import extract_streamer_face
            margin = getattr(cfg.visual_encoder.face_encoder, "crop_margin", 0.2)
            crops, valid_mask = [], []
            for fp in frame_paths:
                frame = cv2.imread(str(fp))
                if frame is not None:
                    cropped, is_tight_face = extract_streamer_face(frame, resolved_bbox, margin=margin)
                    if image_aug is not None:
                        cropped = image_aug(cropped)
                    crops.append(cropped)
                    valid_mask.append(is_tight_face)
            feat, _ = encoders["face"].encode(crops if crops else None, valid_mask=valid_mask)
            features["face"] = feat.squeeze(0)
        else:
            from vie_gameemo.preprocess.face_crop import _tight_face_crop
            logger.info("No webcam for %s — detecting face in full frames", clip_id)
            crops, valid_mask = [], []
            for fp in frame_paths:
                frame = cv2.imread(str(fp))
                if frame is not None:
                    cropped, is_tight_face = _tight_face_crop(frame, fallback=frame)
                    if image_aug is not None:
                        cropped = image_aug(cropped)
                    crops.append(cropped)
                    valid_mask.append(is_tight_face)
            feat, _ = encoders["face"].encode(crops if crops else None, valid_mask=valid_mask)
            features["face"] = feat.squeeze(0)

    if "context" in encoders:
        ctx_enc_type = getattr(cfg.visual_encoder.context_encoder, "type", "vit_imagenet")
        if strategy == "dual_path" and has_face and resolved_bbox is not None:
            from vie_gameemo.preprocess.face_crop import batch_extract_webcam_regions
            if ctx_enc_type == "pose":
                webcam_crops_raw = batch_extract_webcam_regions(
                    frame_paths, resolved_bbox, margin=0.1, resize=False,
                )
                if image_aug is not None:
                    webcam_crops_raw = [image_aug(c) for c in webcam_crops_raw]
                kps_cache_path = (
                    Path(cfg.paths.cache) / "pose_kinematics" / f"{clip_id}_kps.npy"
                )
                features["context"] = encoders["context"].encode(
                    webcam_crops_raw, kps_cache_path=kps_cache_path,
                ).squeeze(0)
            else:
                webcam_crops = batch_extract_webcam_regions(
                    frame_paths, resolved_bbox,
                    target_size=tuple(cfg.visual_encoder.context_encoder.target_size),
                )
                if image_aug is not None:
                    webcam_crops = [image_aug(c) for c in webcam_crops]
                features["context"] = encoders["context"].encode(webcam_crops).squeeze(0)
        else:
            if strategy == "dual_path":
                logger.warning("Webcam not detected for %s — context encoder fallback to full frames", clip_id)
            features["context"] = encoders["context"].encode_from_paths(frame_paths).squeeze(0)

    if "text" in encoders:
        transcript = ann.transcript or ""
        features["text"] = encoders["text"].encode(transcript).squeeze(0)

    features["has_face"] = torch.tensor([has_face], dtype=torch.bool)
    return features


def resolve_webcam_bbox(ann, clip_id: str, webcam_bboxes: dict):
    """Resolve a clip's webcam bbox: annotation field first, then the
    `webcam_bboxes.json` fallback produced by `stage0_preprocess.py`.

    Returns:
        A `WebcamBBox` (or None if no bbox is available from either source).
    """
    resolved_bbox = ann.webcam_bbox
    if resolved_bbox is None and clip_id in webcam_bboxes and webcam_bboxes[clip_id] is not None:
        from vie_gameemo.preprocess.webcam_detector import WebcamBBox as DetectorBBox
        bbox_dict = webcam_bboxes[clip_id]
        resolved_bbox = DetectorBBox(
            xmin=bbox_dict["xmin"], ymin=bbox_dict["ymin"],
            width=bbox_dict["width"], height=bbox_dict["height"],
            stability_score=bbox_dict.get("stability_score", 0.0),
            edge_distance=bbox_dict.get("edge_distance", 0.0),
        )
    return resolved_bbox


@dataclass
class OnlineAugmentContext:
    """Everything needed to re-encode augmented modalities on the fly during
    training. Built once before the training loop (see `build_online_augment_context`).
    """
    encoders: dict                      # subset of {"audio","face","context"} actually online-augmented
    clip_meta: dict[str, tuple]          # clip_id -> (ann, resolved_bbox, has_face, frame_paths)
    strategy: str
    video_index: dict[str, Path]
    image_aug: Callable[[np.ndarray], np.ndarray] | None
    audio_aug: Callable[[np.ndarray, int], np.ndarray] | None
    cfg: object


def build_online_augment_context(cfg, clip_ids: list[str], device, logger: logging.Logger) -> "OnlineAugmentContext | None":
    """Set up online (on-the-fly) raw augmentation, if `augment.raw.mode == "online"`.

    Loads the frozen audio/face/context encoders needed to re-derive
    embeddings from raw data each step — ONLY for modalities that actually
    have an augmentation configured (`augment.raw.image` / `augment.raw.audio`),
    to avoid paying GPU memory/load cost for unused encoders. This is
    substantially slower than cached training (a full encoder forward pass
    per augmented sample per step, instead of a free tensor read) — that's
    the deliberate trade-off online mode makes for genuinely fresh
    augmentation every step, vs `precomputed` mode's fixed K variants.

    Args:
        cfg: Full config namespace.
        clip_ids: All clip_ids that appear in the training split (metadata
            is pre-resolved for exactly these, once, up front).
        device: Torch device for the online encoders.
        logger: Logger for setup/progress messages.

    Returns:
        None if `augment.raw.mode != "online"` (or neither image nor audio
        augmentation is configured) — caller should skip online augmentation
        entirely in that case. Otherwise the built context.
    """
    raw_cfg = getattr(getattr(cfg, "augment", None), "raw", None)
    if raw_cfg is None or getattr(raw_cfg, "mode", "none") != "online":
        return None

    image_aug, audio_aug = build_augment_fns(cfg)
    if image_aug is None and audio_aug is None:
        logger.warning(
            "augment.raw.mode='online' but neither augment.raw.image nor "
            "augment.raw.audio is configured — nothing to augment, disabling."
        )
        return None

    encoders: dict = {}
    if audio_aug is not None:
        from vie_gameemo.encoders import get_audio_encoder
        logger.info("Online augment: loading audio encoder...")
        encoders["audio"] = get_audio_encoder(cfg, device=device)
    if image_aug is not None:
        from vie_gameemo.encoders import get_context_encoder, get_face_encoder
        logger.info("Online augment: loading face + context encoders...")
        encoders["face"] = get_face_encoder(cfg, device=device)
        encoders["context"] = get_context_encoder(cfg).to(device)

    strategy = getattr(cfg.visual_encoder, "strategy", "dual_path")

    video_index: dict[str, Path] = {}
    if "audio" in encoders:
        video_index = index_videos(Path(cfg.paths.raw_videos))

    webcam_bboxes: dict = {}
    if strategy != "full_frame":
        webcam_bbox_file = Path(cfg.paths.processed) / "webcam_bboxes.json"
        if webcam_bbox_file.exists():
            import json
            webcam_bboxes = json.loads(webcam_bbox_file.read_text(encoding="utf-8"))

    from vie_gameemo.data.schemas import Annotation

    annotations_dir = Path(cfg.paths.annotations)
    clip_meta: dict[str, tuple] = {}
    for clip_id in clip_ids:
        ann_path = annotations_dir / f"{clip_id}.json"
        if not ann_path.exists():
            logger.warning("Online augment: no annotation for %s — will skip online re-encoding for it", clip_id)
            continue
        ann = Annotation.load(ann_path)
        resolved_bbox = resolve_webcam_bbox(ann, clip_id, webcam_bboxes)
        has_face = resolved_bbox is not None
        frame_dir = Path(cfg.paths.frames) / clip_id
        frame_paths = sorted(frame_dir.glob("frame_*.jpg")) if frame_dir.exists() else []
        clip_meta[clip_id] = (ann, resolved_bbox, has_face, frame_paths)

    logger.info(
        "Online raw augmentation enabled: re-encoding modalities=%s on the fly "
        "(%d/%d train clips have metadata resolved) — this is slower than "
        "cached training.",
        sorted(encoders.keys()), len(clip_meta), len(clip_ids),
    )

    return OnlineAugmentContext(
        encoders=encoders, clip_meta=clip_meta, strategy=strategy,
        video_index=video_index, image_aug=image_aug, audio_aug=audio_aug, cfg=cfg,
    )


def apply_online_augment(
    clip_ids: list[str], ctx: OnlineAugmentContext, device, logger: logging.Logger,
) -> dict[str, torch.Tensor]:
    """Re-encode this batch's online-augmented modalities from raw data.

    Args:
        clip_ids: Batch's clip_id list (in batch order).
        ctx: Context from `build_online_augment_context`.
        device: Torch device to move the resulting tensors to.
        logger: Logger passed through to `extract_clip_features`.

    Returns:
        Dict mapping each modality in `ctx.encoders` to a freshly stacked
        (B, T, D) tensor — caller should overwrite the corresponding cached
        batch tensors with these before running fusion. A clip_id missing
        metadata (should not normally happen — see `build_online_augment_context`
        warnings) falls back to that modality's own cached zero-shape via a
        zeros tensor matching the batch's other entries, rather than crashing
        mid-batch.
    """
    per_modality: dict[str, list[torch.Tensor]] = {k: [] for k in ctx.encoders}

    for clip_id in clip_ids:
        meta = ctx.clip_meta.get(clip_id)
        if meta is None:
            for k in per_modality:
                per_modality[k].append(None)
            continue
        ann, resolved_bbox, has_face, frame_paths = meta
        feats = extract_clip_features(
            ctx.cfg, ctx.encoders, clip_id, ann, resolved_bbox, has_face, frame_paths,
            ctx.strategy, ctx.video_index, logger,
            image_aug=ctx.image_aug, audio_aug=ctx.audio_aug,
        )
        for k in per_modality:
            per_modality[k].append(feats[k])

    result = {}
    for k, tensors in per_modality.items():
        present = [t for t in tensors if t is not None]
        if not present:
            continue
        fill = torch.zeros_like(present[0])
        stacked = torch.stack([t if t is not None else fill for t in tensors], dim=0)
        result[k] = stacked.to(device)
    return result
