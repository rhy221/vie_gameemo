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

from vie_gameemo.data.dataset import VieGameEmoDataset, collate_fn
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
        "--stage", choices=["perception", "cognition"], required=True,
        help="Which curriculum stage to train",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--fusion", type=str, default=None,
        help="Override fusion type (for ablation)",
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

    train_ds = VieGameEmoDataset(
        annotations_dir=Path(cfg.paths.annotations),
        features_dir=Path(cfg.paths.features),
        split="train",
    )
    val_ds = VieGameEmoDataset(
        annotations_dir=Path(cfg.paths.annotations),
        features_dir=Path(cfg.paths.features),
        split="val",
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
    else:
        if not args.resume_from:
            raise ValueError("--resume-from REQUIRED for cognition stage (perception checkpoint)")
        ckpt = train_cognition(
            cfg=cfg, perception_checkpoint=args.resume_from,
            train_loader=train_loader, val_loader=val_loader,
            device=device,
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
    return overrides


if __name__ == "__main__":
    sys.exit(main())
