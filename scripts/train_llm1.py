"""Train LLM-1 Faithful Explainer (2-stage).

Stage A: alignment (ModalAdapter + g_head only, LLM frozen)
Stage B: LoRA fine-tune (optional, adds LoRA to LLM)

Usage:
    # Stage A only (recommended for small datasets)
    python scripts/train_llm1.py --config config.yaml \\
        --resume-from outputs/checkpoints/perception_best.pt \\
        --stage a

    # Stage B (after Stage A)
    python scripts/train_llm1.py --config config.yaml \\
        --resume-from outputs/checkpoints/perception_best.pt \\
        --stage-a-ckpt outputs/checkpoints/llm1_explanation_best.pt \\
        --stage b

    # Both stages sequentially
    python scripts/train_llm1.py --config config.yaml \\
        --resume-from outputs/checkpoints/perception_best.pt \\
        --stage both

    # Precompute cues only (no training)
    python scripts/train_llm1.py --config config.yaml --precompute-cues
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from vie_gameemo.data.dataset import VieGameEmoDataset
from vie_gameemo.training.llm1_explanation import (
    collate_fn_llm1,
    train_llm1_stage_a,
    train_llm1_stage_b,
)
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLM-1 Faithful Explainer")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument(
        "--stage", choices=["a", "b", "both"], default="a",
        help="Training stage: 'a' (alignment), 'b' (LoRA), 'both' (sequential)",
    )
    parser.add_argument(
        "--resume-from", type=Path, required=False,
        help="Perception checkpoint (required for training)",
    )
    parser.add_argument(
        "--stage-a-ckpt", type=Path, default=None,
        help="Stage A checkpoint for Stage B warm-start",
    )
    parser.add_argument(
        "--precompute-cues", action="store_true",
        help="Only precompute cue cache, no training",
    )
    parser.add_argument("--epochs-a", type=int, default=None)
    parser.add_argument("--epochs-b", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    overrides = {}
    if args.epochs_a is not None:
        overrides["training.llm1_explanation.epochs_a"] = args.epochs_a
    if args.epochs_b is not None:
        overrides["training.llm1_explanation.epochs_b"] = args.epochs_b
    if args.batch_size is not None:
        overrides["training.llm1_explanation.batch_size"] = args.batch_size

    cfg = load_config(args.config, experiment=args.experiment, cli_overrides=overrides)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    # --- Precompute cues ---
    if args.precompute_cues or args.stage in ("a", "both"):
        from vie_gameemo.llm.cue_extractor import CueExtractor

        extractor = CueExtractor(cache_dir=cfg.paths.cache)
        extractor.precompute_all(
            faces_dir=cfg.paths.faces,
            audios_dir=cfg.paths.audios,
            frames_dir=cfg.paths.frames,
        )
        if args.precompute_cues:
            logger.info("Cue precompute done. Exiting.")
            return 0

    if not args.resume_from:
        logger.error("--resume-from (perception checkpoint) is required for training")
        return 1

    device = torch.device(
        cfg.compute.device if cfg.compute.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    logger.info("Device: %s", device)

    # --- Dataset ---
    split_manifest = Path(getattr(cfg.paths, "split_manifest", "data/splits.json"))
    tcfg = cfg.training.llm1_explanation
    batch_size = getattr(tcfg, "batch_size", 8)

    train_ds = VieGameEmoDataset(
        annotations_dir=Path(cfg.paths.annotations),
        features_dir=Path(cfg.paths.features),
        split="train",
        split_manifest=split_manifest if split_manifest.exists() else None,
    )
    val_ds = VieGameEmoDataset(
        annotations_dir=Path(cfg.paths.annotations),
        features_dir=Path(cfg.paths.features),
        split="val",
        split_manifest=split_manifest if split_manifest.exists() else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.compute.num_workers,
        pin_memory=cfg.compute.pin_memory,
        collate_fn=collate_fn_llm1,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.compute.num_workers,
        collate_fn=collate_fn_llm1,
    )

    # --- Training ---
    stage_a_ckpt = args.stage_a_ckpt

    if args.stage in ("a", "both"):
        logger.info("=== Stage A: Alignment ===")
        stage_a_ckpt = train_llm1_stage_a(
            cfg=cfg,
            perception_checkpoint=args.resume_from,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

    if args.stage in ("b", "both"):
        if stage_a_ckpt is None:
            stage_a_ckpt = Path(cfg.paths.checkpoints) / "llm1_explanation_best.pt"
        if not Path(stage_a_ckpt).exists():
            logger.error("Stage A checkpoint not found: %s", stage_a_ckpt)
            return 1

        logger.info("=== Stage B: LoRA fine-tune ===")
        train_llm1_stage_b(
            cfg=cfg,
            perception_checkpoint=args.resume_from,
            stage_a_checkpoint=Path(stage_a_ckpt),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

    logger.info("LLM-1 training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
