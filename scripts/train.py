"""Stage 3-4 (+5 cognition): train curriculum (perception → cognition).

Two stages:
    --stage perception: train fusion + classifier (Stage 1 of curriculum)
    --stage cognition:  joint train LLM + adapter (Stage 2 of curriculum, requires perception checkpoint)

For RLVR (LLM-4), use train_rlvr.py instead.

Usage:
    # Stage 1
    python scripts/train.py --config config.yaml --stage perception

    # Stage 2 (requires perception checkpoint)
    python scripts/train.py --config config.yaml --stage cognition \\
        --resume-from outputs/checkpoints/perception_best.pt

    # With experiment override (e.g., ablation)
    python scripts/train.py --config config.yaml --experiment strategy_b_face_only --stage perception
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from vie_gameemo.data.dataset import VieGameEmoDataset, collate_fn, zero_modalities_for_strategy
from vie_gameemo.training.cognition import train_cognition
from vie_gameemo.training.perception import train_perception
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vie-GameEmo (curriculum)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment override file name (without .yaml) under configs/experiments/",
    )
    parser.add_argument(
        "--stage", choices=["perception", "llm_perception", "cognition"], required=True,
        help="Which curriculum stage to train",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--llm-perception-ckpt", type=Path, default=None,
        help="Stage 2a checkpoint (llm_perception_best.pt) to warm-start cognition from",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--fusion", type=str, default=None,
        help="Override fusion type (for ablation)",
    )
    parser.add_argument(
        "--skip-mlp-if-matched", action="store_true", default=None,
        help=(
            "Ablation: skip a modality's standardizer nn.Linear when its raw "
            "dim already equals fusion.d_model, using nn.Identity() instead. "
            "Only enables (no CLI way to force-disable); omit to use the "
            "config value. Changes the fusion checkpoint shape for matched "
            "modalities — don't mix with checkpoints trained without this flag."
        ),
    )
    parser.add_argument(
        "--zero-modality", action="append", choices=["audio", "face", "context", "text"],
        default=None, dest="zero_modality",
        help=(
            "Ablation: zero out a modality at load time (repeatable, e.g. "
            "--zero-modality audio --zero-modality text). Combined with "
            "whatever visual_encoder.strategy already zeroes (e.g. "
            "face_only always zeroes context)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cli_overrides = _cli_overrides(args)
    cfg = load_config(args.config, experiment=args.experiment, cli_overrides=cli_overrides)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    device = torch.device(cfg.compute.device if cfg.compute.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)
    logger.info("Stage: %s | Fusion: %s | LLM setup: %s",
                args.stage, cfg.fusion.type, cfg.llm.active_setup)

    split_manifest = Path(getattr(cfg.paths, "split_manifest", "data/splits.json"))
    strategy = getattr(getattr(cfg, "visual_encoder", None), "strategy", "dual_path")
    zero_mods = sorted(set(zero_modalities_for_strategy(strategy)) | set(args.zero_modality or []))
    if zero_mods:
        logger.info(
            "Zeroing modalities at load time: %s (strategy='%s' + --zero-modality=%s)",
            zero_mods, strategy, args.zero_modality or [],
        )

    raw_augment_cfg = getattr(getattr(cfg, "augment", None), "raw", None)
    precomputed_copies = 0
    if raw_augment_cfg is not None and getattr(raw_augment_cfg, "mode", "none") == "precomputed":
        precomputed_copies = getattr(raw_augment_cfg, "precomputed_copies", 5)

    train_ds = VieGameEmoDataset(
        annotations_dir=Path(cfg.paths.annotations),
        features_dir=Path(cfg.paths.features),
        split="train",
        split_manifest=split_manifest if split_manifest.exists() else None,
        zero_modalities=zero_mods,
        precomputed_augment_copies=precomputed_copies,
    )
    val_ds = VieGameEmoDataset(
        annotations_dir=Path(cfg.paths.annotations),
        features_dir=Path(cfg.paths.features),
        split="val",
        split_manifest=split_manifest if split_manifest.exists() else None,
        zero_modalities=zero_mods,
        precomputed_augment_copies=precomputed_copies,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.perception.batch_size if args.stage == "perception"
                   else cfg.training.cognition.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
        pin_memory=cfg.compute.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=cfg.compute.num_workers, collate_fn=collate_fn,
    )

    if args.stage == "perception":
        ckpt = train_perception(
            cfg=cfg, train_loader=train_loader, val_loader=val_loader,
            device=device, resume_from=args.resume_from,
        )
    elif args.stage == "llm_perception":
        if not args.resume_from:
            raise ValueError("--resume-from REQUIRED for llm_perception stage (perception checkpoint)")
        from vie_gameemo.training.cognition import train_llm_perception
        use_hint = cfg.llm.active_setup in ("llm2",)
        ckpt = train_llm_perception(
            cfg=cfg, perception_checkpoint=args.resume_from,
            train_loader=train_loader, val_loader=val_loader,
            device=device, use_mlp_hint=use_hint,
        )
    else:
        if not args.resume_from:
            raise ValueError("--resume-from REQUIRED for cognition stage (perception checkpoint)")
        ckpt = train_cognition(
            cfg=cfg, perception_checkpoint=args.resume_from,
            train_loader=train_loader, val_loader=val_loader,
            device=device,
            llm_perception_checkpoint=args.llm_perception_ckpt,
        )

    logger.info("Training complete. Best checkpoint: %s", ckpt)
    return 0


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Build dict of dot-path overrides from non-None CLI args."""
    overrides = {}
    if args.epochs is not None:
        overrides[f"training.{args.stage}.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides[f"training.{args.stage}.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides[f"training.{args.stage}.learning_rate.fusion"] = args.lr
    if args.fusion is not None:
        overrides["fusion.type"] = args.fusion
    if args.skip_mlp_if_matched:
        overrides["fusion.skip_mlp_if_matched"] = True
    return overrides


if __name__ == "__main__":
    sys.exit(main())
