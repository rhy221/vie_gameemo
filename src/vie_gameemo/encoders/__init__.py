"""Stage 2 encoders for the 4 modalities.

Modules:
    - audio_whisper: Whisper encoder for speech prosody features
    - audio_ast: AST encoder for non-speech audio events (ablation)
    - audio_wav2vec2: Wav2Vec2 self-supervised speech encoder (ablation)
    - audio_hubert: HuBERT self-supervised speech encoder (ablation)
    - face_vit: ViT-FER for streamer face (Path 1 of dual-path)
    - context_vit: ViT-ImageNet for gameplay context (Path 2 — vit_imagenet branch)
    - context_pose: Pose-kinematics encoder (Path 2 — pose branch, Phase 0)
    - text_xlmr: XLM-RoBERTa for transcript

All encoders are FROZEN during training (only fusion + classifier are trained).
Outputs are token sequences of shape (B, T, d_out) ready for fusion, where
d_out defaults to each encoder's native hidden size — see
`vie_gameemo.fusion.modality_dim_kwargs` for how dimension mismatches across
audio backbones are standardized (in the trainable fusion MLP, not here).

Face encoder selection:
    Use ``get_face_encoder(cfg, device)`` to get the right encoder per config:
      - ``visual_encoder.face_encoder.backend = "vit"`` (default) → trpakov/vit-face-expression
      - ``visual_encoder.face_encoder.backend = "vit_multi"``     → mo-thecreator/vit-Facial-Expression-Recognition

Context encoder selection:
    Use ``get_context_encoder(cfg)`` to get the right encoder per config:
      - ``visual_encoder.context_encoder.type = "vit_imagenet"`` → ContextEncoder
        (backbone chosen via ``visual_encoder.context_encoder.backend``:
        "vit" (default, google/vit-base-patch16-224) | "eva_vit_b" (EVA ViT-B,
        loaded through transformers' timm-wrapper — requires `timm` installed))
      - ``visual_encoder.context_encoder.type = "pose"``        → PoseContextEncoder

Audio encoder selection:
    Use ``get_audio_encoder(cfg, device)`` to get the right encoder per config:
      - ``audio_encoder.type = "whisper"``   → WhisperAudioEncoder (default)
      - ``audio_encoder.type = "ast"``       → ASTAudioEncoder
      - ``audio_encoder.type = "wav2vec2"``  → Wav2Vec2AudioEncoder
      - ``audio_encoder.type = "hubert"``    → HubertAudioEncoder
"""

import torch
import torch.nn as nn


def get_context_encoder(cfg) -> nn.Module:
    """Factory: instantiate context encoder per ``visual_encoder.context_encoder.type``.

    Args:
        cfg: Project config namespace (supports both SimpleNamespace and OmegaConf).

    Returns:
        Frozen context encoder module with an ``encode_batch`` method.
    """
    try:
        ctx_cfg = cfg.visual_encoder.context_encoder
    except AttributeError:
        ctx_cfg = None

    enc_type = getattr(ctx_cfg, "type", "vit_imagenet") if ctx_cfg is not None else "vit_imagenet"

    if enc_type == "pose":
        from vie_gameemo.encoders.context_pose import PoseContextEncoder

        pose_cfg = getattr(ctx_cfg, "pose_temporal", None)
        hidden_dim = getattr(pose_cfg, "hidden_dim", 128) if pose_cfg else 128
        n_layers = getattr(pose_cfg, "n_layers", 1) if pose_cfg else 1
        dropout = getattr(pose_cfg, "dropout", 0.3) if pose_cfg else 0.3

        # P0.0-fix.6: n_frames_pose is separate from ViT's n_frames so pose sampling
        # can be calibrated independently (e.g. sample denser for impulse events).
        # Falls back to n_frames for now; calibrate at P0.3.
        n_frames_pose = getattr(ctx_cfg, "n_frames_pose", getattr(ctx_cfg, "n_frames", 16))
        return PoseContextEncoder(
            backend=getattr(ctx_cfg, "pose_backend", "mediapipe"),
            n_frames=n_frames_pose,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
    else:  # vit_imagenet (default / ablation)
        from vie_gameemo.encoders.context_vit import ContextEncoder

        backend = getattr(ctx_cfg, "backend", "vit") if ctx_cfg is not None else "vit"
        model_name = resolve_context_vit_model_name(ctx_cfg, backend)

        return ContextEncoder(
            model_name=model_name,
            backend=backend,
            n_frames=getattr(ctx_cfg, "n_frames", 16),
            target_size=tuple(getattr(ctx_cfg, "target_size", (224, 224))),
            temporal_pool=getattr(ctx_cfg, "temporal_pool", "mean"),
            use_patch_tokens=getattr(ctx_cfg, "use_patch_tokens", False),
            spatial_pool=tuple(getattr(ctx_cfg, "spatial_pool", (2, 2))),
        )


_CONTEXT_VIT_DEFAULT_MODEL = {
    "vit": "google/vit-base-patch16-224",
    "eva_vit_b": "timm/eva02_base_patch14_224.mim_in22k",
}


def resolve_context_vit_model_name(ctx_cfg, backend: str) -> str:
    """Resolve the checkpoint for the active context-encoder ``backend``.

    Mirrors `resolve_audio_model_name`: reads ``context_encoder.models.<backend>``
    first, then falls back to the flat ``model_name`` field (so existing configs
    that only set `model_name` for the "vit" backend keep working unchanged),
    then to a built-in default per backend.
    """
    models_cfg = getattr(ctx_cfg, "models", None) if ctx_cfg is not None else None
    model_name = getattr(models_cfg, backend, None) if models_cfg is not None else None
    return (
        model_name
        or (getattr(ctx_cfg, "model_name", None) if ctx_cfg is not None else None)
        or _CONTEXT_VIT_DEFAULT_MODEL.get(backend, _CONTEXT_VIT_DEFAULT_MODEL["vit"])
    )


_FACE_VIT_DEFAULT_MODEL = {
    "vit": "trpakov/vit-face-expression",
    "vit_multi": "mo-thecreator/vit-Facial-Expression-Recognition",
}


def resolve_face_model_name(face_cfg, backend: str) -> str:
    """Resolve the checkpoint for the active face-encoder ``backend``.

    Mirrors `resolve_context_vit_model_name`: reads ``face_encoder.models.<backend>``
    first, then falls back to the flat ``model_name`` field, then to a built-in
    default per backend.
    """
    models_cfg = getattr(face_cfg, "models", None) if face_cfg is not None else None
    model_name = getattr(models_cfg, backend, None) if models_cfg is not None else None
    return (
        model_name
        or (getattr(face_cfg, "model_name", None) if face_cfg is not None else None)
        or _FACE_VIT_DEFAULT_MODEL.get(backend, _FACE_VIT_DEFAULT_MODEL["vit"])
    )


def get_face_encoder(cfg, device: str | torch.device = "cuda") -> nn.Module:
    """Factory: instantiate face encoder per ``visual_encoder.face_encoder.backend``.

    Args:
        cfg: Project config namespace (supports both SimpleNamespace and OmegaConf).
        device: Torch device for the encoder.

    Returns:
        Frozen `FaceEncoder` (ViT-FER, Path 1 of dual-path).
    """
    from vie_gameemo.encoders.face_vit import FaceEncoder

    try:
        face_cfg = cfg.visual_encoder.face_encoder
    except AttributeError:
        face_cfg = None

    backend = getattr(face_cfg, "backend", "vit") if face_cfg is not None else "vit"
    model_name = resolve_face_model_name(face_cfg, backend)

    temporal_cfg = getattr(getattr(face_cfg, "dual_view", None), "temporal", None)
    n_temporal_frames = getattr(temporal_cfg, "n_frames", 16) if temporal_cfg else 16
    # NOTE: spatial_pool intentionally NOT read from dual_view.temporal.spatial_pool
    # here — existing call sites never passed it either (always used FaceEncoder's
    # (4, 4) default), so wiring it in now would silently change n_patch_tokens
    # (and thus total_tokens / cached feature shapes) for anyone already relying
    # on the current default. Left as a known gap, not fixed here.

    # dual_view.global.source selects peak-frame/global-CLS strategy — see
    # FaceEncoder's peak_frame_source docstring. "auto_peak" (default, current
    # behavior since commit fef0294) if unset; set to "middle_frame" in
    # config.yaml to reproduce the pre-fef0294 behavior for checkpoints
    # trained before that change.
    global_cfg = getattr(getattr(face_cfg, "dual_view", None), "global", None)
    peak_frame_source = getattr(global_cfg, "source", "auto_peak") if global_cfg else "auto_peak"

    return FaceEncoder(
        model_name=model_name,
        backend=backend,
        n_temporal_frames=n_temporal_frames,
        target_size=tuple(getattr(face_cfg, "target_size", (224, 224)) if face_cfg is not None else (224, 224)),
        peak_frame_source=peak_frame_source,
        device=device,
    )


_AUDIO_ENCODER_DEFAULT_MODEL = {
    "whisper": "openai/whisper-small",
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
}


def resolve_audio_model_name(cfg) -> str:
    """Resolve the checkpoint to load for the active ``audio_encoder.type``.

    Reads ``audio_encoder.models.<type>`` first — a per-type mapping, e.g.::

        audio_encoder:
          type: "wav2vec2"
          models:
            whisper: "openai/whisper-small"
            wav2vec2: "facebook/wav2vec2-base"

    This means ablating across types only requires flipping ``type``; a flat
    ``audio_encoder.model_name`` (old-style, single field shared by all types)
    would silently keep pointing at e.g. a Whisper checkpoint after switching
    ``type`` to "ast" unless also updated by hand. Falls back to the flat
    ``model_name`` field, then to a built-in default per type.
    """
    acfg = cfg.audio_encoder
    enc_type = getattr(acfg, "type", "whisper")
    models_cfg = getattr(acfg, "models", None)
    model_name = getattr(models_cfg, enc_type, None) if models_cfg is not None else None
    return (
        model_name
        or getattr(acfg, "model_name", None)
        or _AUDIO_ENCODER_DEFAULT_MODEL[enc_type]
    )


def get_audio_encoder(cfg, device: str | torch.device = "cuda") -> nn.Module:
    """Factory: instantiate audio encoder per ``audio_encoder.type``.

    Args:
        cfg: Project config namespace (supports both SimpleNamespace and OmegaConf).
        device: Torch device for the encoder.

    Returns:
        Frozen audio encoder module with ``encode`` / ``encode_batch`` methods
        and a ``d_out`` attribute reporting its actual output dim (see
        `vie_gameemo.fusion.modality_dim_kwargs` to standardize across types).

    Raises:
        KeyError: If ``audio_encoder.type`` is not one of
            "whisper" | "ast" | "wav2vec2" | "hubert".
    """
    acfg = cfg.audio_encoder
    enc_type = getattr(acfg, "type", "whisper")

    if enc_type not in _AUDIO_ENCODER_DEFAULT_MODEL:
        raise KeyError(
            f"Unknown audio_encoder.type '{enc_type}'. "
            f"Available: {sorted(_AUDIO_ENCODER_DEFAULT_MODEL)}"
        )

    model_name = resolve_audio_model_name(cfg)
    target_tokens = getattr(acfg, "target_tokens", 64)
    d_out = getattr(acfg, "d_out", None)
    sample_rate = getattr(getattr(cfg, "preprocess", None), "audio", None)
    sample_rate = getattr(sample_rate, "sample_rate", 16000) if sample_rate is not None else 16000

    common_kwargs = dict(
        model_name=model_name,
        target_tokens=target_tokens,
        d_out=d_out,
        sample_rate=sample_rate,
        device=device,
    )

    if enc_type == "whisper":
        from vie_gameemo.encoders.audio_whisper import WhisperAudioEncoder
        return WhisperAudioEncoder(**common_kwargs)
    elif enc_type == "ast":
        from vie_gameemo.encoders.audio_ast import ASTAudioEncoder
        return ASTAudioEncoder(**common_kwargs)
    elif enc_type == "wav2vec2":
        from vie_gameemo.encoders.audio_wav2vec2 import Wav2Vec2AudioEncoder
        return Wav2Vec2AudioEncoder(**common_kwargs)
    else:  # hubert
        from vie_gameemo.encoders.audio_hubert import HubertAudioEncoder
        return HubertAudioEncoder(**common_kwargs)
