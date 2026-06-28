"""Multi-agent annotation orchestrator.

Coordinates the per-clip annotation flow:
    peak_frame → openface_au → qwen_vl → qwen_audio → whisper → consolidator

Memory-conscious: each agent is loaded, used for the entire batch, then
unloaded before the next agent loads. This serial pattern fits on consumer
GPUs (T4, L4).

Output: one JSON file per clip under data/annotations/, validated by the
Annotation schema.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from vie_gameemo.data.annotator.consolidator import Consolidator
from vie_gameemo.data.annotator.openface_au import extract_aus
from vie_gameemo.data.annotator.peak_frame import detect_peak_frame, extract_frame_at_index
from vie_gameemo.data.annotator.qwen_audio_agent import QwenAudioAgent
from vie_gameemo.data.annotator.qwen_vl_agent import QwenVLAgent
from vie_gameemo.data.annotator.whisper_asr import WhisperASR
from vie_gameemo.data.schemas import Annotation, AnnotatorAgent, WebcamBBox, EmotionLabel

logger = logging.getLogger(__name__)


def annotate_batch(
    clip_paths: list[Path],
    emotion_labels: list[str],
    output_dir: Path,
    cfg: SimpleNamespace,
    resume: bool = True,
) -> list[Path]:
    """Annotate a batch of clips with the full multi-agent pipeline.

    Phases (each phase loads its model, processes all clips, then unloads):
        1. Local CPU/GPU: OpenFace AU + peak frame extraction + Whisper ASR
        2. Qwen2.5-VL: visual objective description
        3. Qwen2-Audio: audio prosody description
        4. Qwen2.5-Instruct (consolidator): merge into reasoning

    Args:
        clip_paths: Video file paths.
        emotion_labels: Ground-truth labels (1-to-1 with clip_paths).
        output_dir: Where to save per-clip Annotation JSONs.
        cfg: Annotation config namespace (config.annotation).
        resume: If True, skip clips with existing annotation files.

    Returns:
        Paths to annotation JSON files (one per clip).

    Raises:
        ValueError: If clip_paths and emotion_labels have different lengths.
    """
    if len(clip_paths) != len(emotion_labels):
        raise ValueError(
            f"clip_paths ({len(clip_paths)}) and emotion_labels "
            f"({len(emotion_labels)}) must have the same length"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter already-annotated clips if resume mode
    pending_indices: list[int] = []
    result_paths: list[Path | None] = [None] * len(clip_paths)

    for i, clip_path in enumerate(clip_paths):
        ann_path = output_dir / f"{clip_path.stem}.json"
        if resume and ann_path.exists():
            logger.info("Skip (already annotated): %s", clip_path.name)
            result_paths[i] = ann_path
        else:
            pending_indices.append(i)

    if not pending_indices:
        logger.info("All clips already annotated")
        return [p for p in result_paths if p is not None]

    pending_clips = [clip_paths[i] for i in pending_indices]
    pending_labels = [emotion_labels[i] for i in pending_indices]

    logger.info("Annotating %d clips (skipping %d already done)",
                len(pending_clips), len(clip_paths) - len(pending_clips))

    # Phase 1: Lightweight local ops
    t0 = time.time()
    local_results = _phase_local(pending_clips, cfg)
    logger.info("Phase 1 done in %.1fs", time.time() - t0)

    peak_frame_paths = [r["peak_frame_path"] for r in local_results]
    webcam_crops = [r.get("face_crop") for r in local_results]
    audio_paths = [r["audio_path"] for r in local_results]

    # Phase 2: Visual descriptions from webcam region
    t0 = time.time()
    visual_descs = _phase_visual_descriptions(webcam_crops, peak_frame_paths, cfg)
    logger.info("Phase 2 (VL webcam) done in %.1fs", time.time() - t0)

    # Phase 3: Audio descriptions
    t0 = time.time()
    audio_descs = _phase_audio_descriptions(audio_paths, cfg)
    logger.info("Phase 3 (Audio) done in %.1fs", time.time() - t0)

    # Phase 4: Consolidation
    t0 = time.time()
    consolidation_inputs = [
        {
            "emotion_label": pending_labels[i],
            "face_aus": local_results[i]["face_aus"],
            "face_description": visual_descs[i],
            "visual_objective": "",
            "audio_tone": audio_descs[i],
            "transcript": local_results[i]["transcript"],
        }
        for i in range(len(pending_clips))
    ]
    reasonings = _phase_consolidate(consolidation_inputs, cfg)
    logger.info("Phase 4 (Consolidate) done in %.1fs", time.time() - t0)

    # Compose and save Annotation objects
    for idx, i in enumerate(pending_indices):
        clip_path = clip_paths[i]
        ann_path = output_dir / f"{clip_path.stem}.json"

        lr = local_results[idx]
        annotation = Annotation(
            clip_id=clip_path.stem,
            emotion_label=EmotionLabel(pending_labels[idx]),
            genre=_infer_genre(clip_path, cfg),
            face_aus={str(k): float(v) for k, v in lr["peak_frame_aus"].items()},
            peak_frame_idx=lr["peak_frame_idx"],
            webcam_bbox=lr.get("webcam_bbox"),
            face_description=visual_descs[idx],
            visual_objective_desc="",
            audio_tone_desc=audio_descs[idx],
            transcript=lr["transcript"],
            reasoning=reasonings[idx],
            annotators=[
                AnnotatorAgent(agent_name="openface", model_version="2.2.0", output=""),
                AnnotatorAgent(agent_name="qwen_vl_webcam", model_version=cfg.qwen_vl.model_name, output=visual_descs[idx]),
                AnnotatorAgent(agent_name="qwen_audio", model_version=cfg.qwen_audio.model_name, output=audio_descs[idx]),
                AnnotatorAgent(agent_name="whisper", model_version=getattr(cfg, 'asr', cfg).whisper.model_name if hasattr(cfg, 'asr') else "whisper", output=lr["transcript"]),
                AnnotatorAgent(agent_name="consolidator", model_version=cfg.consolidator.model_name, output=reasonings[idx]),
            ],
            created_at=datetime.now(timezone.utc),
        )
        annotation.save(ann_path)
        result_paths[i] = ann_path
        logger.info("Saved annotation: %s", ann_path.name)

    return [p for p in result_paths if p is not None]


def _phase_local(
    clip_paths: list[Path],
    cfg: SimpleNamespace,
) -> list[dict]:
    """Phase 1: lightweight ops (OpenFace, peak frame, ASR).

    Args:
        clip_paths: Video files to process.
        cfg: Annotation config (cfg.asr.backend selects ASR backend).

    Returns:
        List of dicts with: peak_frame_idx, peak_frame_path, face_aus,
            peak_frame_aus, audio_path, transcript, webcam_bbox.
    """
    from vie_gameemo.data.annotator.whisper_asr import build_asr

    asr_inst, bartpho_inst = build_asr(cfg.asr)
    asr_inst.load()
    if bartpho_inst is not None:
        bartpho_inst.load()

    results = []
    try:
        for clip_path in clip_paths:
            r = _process_single_clip_local(clip_path, cfg, asr_inst, bartpho_inst)
            results.append(r)
    finally:
        asr_inst.unload()
        if bartpho_inst is not None:
            bartpho_inst.unload()

    return results


def _process_single_clip_local(
    clip_path: Path,
    cfg: SimpleNamespace,
    asr_inst,
    bartpho_inst,
) -> dict:
    """Run local processing for one clip: OpenFace + peak frame + ASR.

    Args:
        clip_path: Video file.
        cfg: Annotation config.
        asr_inst: Already-loaded ASR instance (WhisperASR or PhoWhisperASR).
        bartpho_inst: Already-loaded BARTphoPostProcessor or None.

    Returns:
        Dict with processing results.
    """
    from vie_gameemo.preprocess.demux import extract_audio
    from vie_gameemo.preprocess.webcam_detector import WebcamDetector

    # Extract audio
    audio_dir = Path(str(cfg).split("'")[0]) if hasattr(cfg, "__dict__") else Path("data/processed/audios")
    # Use clip stem as audio filename
    audio_path = Path("data") / "processed" / "audios" / f"{clip_path.stem}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        extract_audio(clip_path, audio_path)
    except Exception as exc:
        logger.warning("Audio extraction failed for %s: %s", clip_path.name, exc)

    # OpenFace AU extraction
    openface_binary = Path(cfg.openface.binary_path) if hasattr(cfg.openface, "binary_path") else None
    au_intensities: dict = {}
    if openface_binary and openface_binary.exists():
        try:
            au_intensities = extract_aus(
                clip_path,
                openface_binary=openface_binary,
                output_dir=Path("data") / "cache" / "openface",
                target_aus=cfg.peak_frame.target_aus if hasattr(cfg.peak_frame, "target_aus") else None,
            )
        except Exception as exc:
            logger.warning("OpenFace failed for %s: %s", clip_path.name, exc)
    else:
        logger.warning("OpenFace binary not available; skipping AU extraction for %s", clip_path.name)

    # Peak frame detection
    method = cfg.peak_frame.method if hasattr(cfg.peak_frame, "method") else "middle"
    if not au_intensities:
        method = "middle"
    peak_idx, _ = detect_peak_frame(clip_path, au_intensities or None, method=method)

    # Save peak frame as image
    peak_frame_dir = Path("data") / "cache" / "peak_frames"
    peak_frame_dir.mkdir(parents=True, exist_ok=True)
    peak_frame_path = peak_frame_dir / f"{clip_path.stem}_peak.jpg"
    if not peak_frame_path.exists():
        try:
            frame_img = extract_frame_at_index(clip_path, peak_idx)
            frame_img.save(str(peak_frame_path), quality=90)
        except Exception as exc:
            logger.warning("Peak frame extraction failed: %s", exc)

    # Per-frame AU intensities at peak
    peak_aus: dict = {}
    if au_intensities:
        for au_code, intensities in au_intensities.items():
            if peak_idx < len(intensities):
                peak_aus[str(au_code)] = intensities[peak_idx]

    # Webcam detection
    webcam_bbox = None
    try:
        detector = WebcamDetector(
            backend="owlv2",
            min_detection_confidence=0.1,
            sample_n_frames=30,
            stability_threshold=0.3,
            dbscan_eps=0.08,
            dbscan_min_samples=3,
        )
        raw_bbox = detector.detect_webcam_region(clip_path)
        if raw_bbox is not None:
            webcam_bbox = WebcamBBox(
                xmin=raw_bbox.xmin,
                ymin=raw_bbox.ymin,
                width=raw_bbox.width,
                height=raw_bbox.height,
                stability_score=raw_bbox.stability_score,
            )
    except Exception as exc:
        logger.warning("Webcam detection failed for %s: %s", clip_path.name, exc)

    # ASR transcription + optional BARTpho post-processing
    transcript = ""
    if audio_path.exists():
        try:
            transcript = asr_inst.transcribe(audio_path)
            if bartpho_inst is not None and transcript:
                transcript = bartpho_inst.process(transcript)
        except Exception as exc:
            logger.warning("ASR failed for %s: %s", clip_path.name, exc)

    # Face crop from webcam region (for VLM face description)
    face_crop = None
    if webcam_bbox is not None and peak_frame_path.exists():
        try:
            import cv2
            from vie_gameemo.preprocess.face_crop import extract_streamer_face
            from vie_gameemo.preprocess.webcam_detector import WebcamBBox as WBBox
            frame = cv2.imread(str(peak_frame_path))
            if frame is not None:
                wb = WBBox(
                    xmin=webcam_bbox.xmin, ymin=webcam_bbox.ymin,
                    width=webcam_bbox.width, height=webcam_bbox.height,
                    stability_score=webcam_bbox.stability_score,
                    edge_distance=0.0,
                )
                face_crop = extract_streamer_face(frame, wb, target_size=(224, 224), margin=0.2)
        except Exception as exc:
            logger.debug("Face crop for VLM failed: %s", exc)

    return {
        "peak_frame_idx": peak_idx,
        "peak_frame_path": peak_frame_path,
        "face_aus": au_intensities,
        "peak_frame_aus": peak_aus,
        "audio_path": audio_path,
        "transcript": transcript,
        "webcam_bbox": webcam_bbox,
        "face_crop": face_crop,
    }


def _phase_visual_descriptions(
    webcam_crops: list,
    peak_frame_paths: list[Path],
    cfg: SimpleNamespace,
) -> list[str]:
    """Phase 2: Qwen2.5-VL describes streamer from webcam region.

    Uses webcam crop when available, falls back to full peak frame.

    Args:
        webcam_crops: List of numpy arrays (BGR) or None per clip.
        peak_frame_paths: Fallback full-frame paths.
        cfg: Annotation config.

    Returns:
        List of Vietnamese descriptions.
    """
    agent = QwenVLAgent(
        model_name=cfg.qwen_vl.model_name,
        prompt=cfg.qwen_vl.prompt,
        quantization=cfg.qwen_vl.quantization,
        max_new_tokens=cfg.qwen_vl.max_new_tokens,
        temperature=cfg.qwen_vl.temperature,
    )
    try:
        agent.load()
        # Use webcam crop where available, fallback to peak frame
        from PIL import Image
        import cv2
        images = []
        for crop, frame_path in zip(webcam_crops, peak_frame_paths):
            if crop is not None:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb))
            elif frame_path.exists():
                images.append(Image.open(frame_path).convert("RGB"))
            else:
                images.append(None)
        results = agent.batch_describe_images(images)
    finally:
        agent.unload()
    return results


def _phase_audio_descriptions(
    audio_paths: list[Path],
    cfg: SimpleNamespace,
) -> list[str]:
    """Phase 3: Qwen2-Audio describes each audio. Load → batch → unload.

    Args:
        audio_paths: Paths to wav files.
        cfg: Annotation config.

    Returns:
        List of Vietnamese prosody descriptions.
    """
    agent = QwenAudioAgent(
        model_name=cfg.qwen_audio.model_name,
        prompt=cfg.qwen_audio.prompt,
        quantization=cfg.qwen_audio.quantization,
        max_new_tokens=cfg.qwen_audio.max_new_tokens,
    )
    try:
        agent.load()
        results = agent.batch_describe(audio_paths)
    finally:
        agent.unload()
    return results


def _phase_consolidate(
    inputs: list[dict],
    cfg: SimpleNamespace,
) -> list[str]:
    """Phase 4: consolidator merges all per-modality descriptions.

    Args:
        inputs: List of dicts with consolidator inputs.
        cfg: Annotation config.

    Returns:
        List of reasoning strings.
    """
    cons = Consolidator(
        model_name=cfg.consolidator.model_name,
        quantization=cfg.consolidator.quantization,
        max_new_tokens=cfg.consolidator.max_new_tokens,
        temperature=cfg.consolidator.temperature,
    )
    try:
        cons.load()
        results = cons.batch_consolidate(inputs)
    finally:
        cons.unload()
    return results


def _infer_genre(clip_path: Path, cfg: SimpleNamespace) -> str:
    """Infer game genre from clip filename or default to 'moba'.

    Args:
        clip_path: Path to clip.
        cfg: Annotation config.

    Returns:
        Genre string.
    """
    from vie_gameemo.data.schemas import GameGenre
    name_lower = clip_path.stem.lower()
    for genre in GameGenre:
        if genre.value in name_lower:
            return genre.value
    return GameGenre.MOBA.value
