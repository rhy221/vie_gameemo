"""Stage 2 encoders for the 4 modalities.

Modules:
    - audio_whisper: Whisper encoder for speech prosody features
    - face_vit: ViT-FER for streamer face (Path 1 of dual-path)
    - context_vit: ViT-ImageNet for gameplay context (Path 2 — vit_imagenet branch)
    - context_eva02: EVA-02-B for gameplay context (Path 2 — eva02_b branch)
    - context_pose: Pose-kinematics encoder (Path 2 — pose branch, Phase 0)
    - text_xlmr: XLM-RoBERTa for transcript

All encoders are FROZEN during training (only fusion + classifier are trained).
Outputs are token sequences of shape (B, T, 768) ready for fusion.

Context encoder selection:
    Use ``get_context_encoder(cfg)`` to get the right encoder per config:
      - ``visual_encoder.context_encoder.type = "vit_imagenet"`` → ContextEncoder
      - ``visual_encoder.context_encoder.type = "eva02_b"``      → Eva02ContextEncoder
      - ``visual_encoder.context_encoder.type = "pose"``         → PoseContextEncoder
"""

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
    elif enc_type == "eva02_b":
        from vie_gameemo.encoders.context_eva02 import Eva02ContextEncoder

        # Config lives under context_encoder.eva02_b sub-key; fall back to flat
        # ctx_cfg keys so a minimal config (type only) still works.
        eva_cfg = getattr(ctx_cfg, "eva02_b", ctx_cfg)
        return Eva02ContextEncoder(
            model_name=getattr(eva_cfg, "model_name", "eva02_base_patch14_224.mim_in22k"),
            n_frames=getattr(eva_cfg, "n_frames", 16),
            spatial_pool=tuple(getattr(eva_cfg, "spatial_pool", [2, 2])),
            temporal_spatial_pool=tuple(getattr(eva_cfg, "temporal_spatial_pool", [2, 2])),
            pool_method=getattr(eva_cfg, "pool_method", "mean"),
            target_size=tuple(getattr(eva_cfg, "target_size", (224, 224))),
            use_temporal_3d_pool=getattr(eva_cfg, "use_temporal_3d_pool", False),
            temporal_3d_pool=tuple(getattr(eva_cfg, "temporal_3d_pool", [4, 4, 4])),
        )

    else:  # vit_imagenet (default / ablation)
        from vie_gameemo.encoders.context_vit import ContextEncoder

        raw_pool = getattr(ctx_cfg, "spatial_pool", [2, 2])
        raw_temporal_pool = getattr(ctx_cfg, "temporal_spatial_pool", [2, 2])
        return ContextEncoder(
            model_name=getattr(ctx_cfg, "model_name", "google/vit-base-patch16-224"),
            n_frames=getattr(ctx_cfg, "n_frames", 16),
            spatial_pool=tuple(raw_pool),
            temporal_spatial_pool=tuple(raw_temporal_pool),
            pool_method=getattr(ctx_cfg, "pool_method", "mean"),
            target_size=tuple(getattr(ctx_cfg, "target_size", (224, 224))),
            temporal_pool=getattr(ctx_cfg, "temporal_pool", None),
            use_temporal_3d_pool=getattr(ctx_cfg, "use_temporal_3d_pool", False),
            temporal_3d_pool=tuple(getattr(ctx_cfg, "temporal_3d_pool", [4, 4, 4])),
        )
