"""Evaluate a trained checkpoint.

Computes metrics on val/test_id/test_ood splits, optionally with per-genre
breakdown and reasoning evaluation.

Usage:
    # Standard eval
    python scripts/eval.py --config config.yaml --checkpoint outputs/checkpoints/perception_best.pt

    # On OOD split
    python scripts/eval.py --config config.yaml --checkpoint ... --split test_ood

    # Strategy A vs B vs C ablation
    python scripts/eval.py --config config.yaml --ablation strategy

    # Per-genre breakdown
    python scripts/eval.py --config config.yaml --checkpoint ... --per-genre
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.io import write_json
from vie_gameemo.utils.logging import setup_logging
from vie_gameemo.utils.seed import set_seed

logger = logging.getLogger(__name__)

_EMOTION_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]
_LABEL2IDX = {l: i for i, l in enumerate(_EMOTION_LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained checkpoint (required unless --ablation)")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "test_id", "test_ood"])
    parser.add_argument("--per-genre", action="store_true",
                        help="Add per-genre breakdown")
    parser.add_argument("--include-reasoning", action="store_true",
                        help="Evaluate LLM reasoning quality (slower)")
    parser.add_argument(
        "--ablation", choices=["strategy", "fusion", "llm"], default=None,
        help="Run an ablation suite instead of single-checkpoint eval",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/results/eval.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)

    if args.ablation:
        results = _run_ablation(cfg, args)
    else:
        if not args.checkpoint:
            ckpt_dir = Path(cfg.paths.checkpoints)
            candidates = sorted(ckpt_dir.glob("perception_best*.pt"), reverse=True)
            if not candidates:
                candidates = sorted(ckpt_dir.glob("*.pt"), reverse=True)
            if candidates:
                args.checkpoint = candidates[0]
                logger.info("Auto-detected checkpoint: %s", args.checkpoint)
            else:
                raise ValueError(
                    f"--checkpoint required (no .pt found in {ckpt_dir})"
                )
        results = _run_eval(cfg, args)

    write_json(results, args.output)
    logger.info("Results saved → %s", args.output)
    return 0


def _run_eval(cfg, args) -> dict:
    """Standard checkpoint eval on a single split."""
    from torch.utils.data import DataLoader

    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.data.dataset import VieGameEmoDataset, collate_fn
    from vie_gameemo.evaluation.metrics import compute_metrics
    from vie_gameemo.evaluation.per_genre import per_genre_metrics
    from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
    from vie_gameemo.training.perception import (
        infer_fusion_dims_from_checkpoint,
        load_checkpoint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fcfg = cfg.fusion
    ccfg = cfg.classifier

    # Build fusion to match the checkpoint's actual architecture. The checkpoint
    # is the source of truth for per-modality dims (text_dim, etc.); fall back
    # to config when the checkpoint doesn't pin them down.
    dim_kwargs = modality_dim_kwargs(fcfg)
    dim_kwargs.update(infer_fusion_dims_from_checkpoint(args.checkpoint, fcfg.d_model))
    fusion = get_fusion(
        fcfg.type,
        d_model=fcfg.d_model,
        n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
        **dim_kwargs,
    ).to(device)
    classifier = EmotionClassifier(
        d_model=fcfg.d_model,
        hidden_dim=ccfg.hidden_dim,
        n_classes=ccfg.n_classes,
        dropout=ccfg.dropout,
    ).to(device)
    load_checkpoint(args.checkpoint, fusion, classifier)
    fusion.eval()
    classifier.eval()

    annotations_dir = Path(cfg.paths.annotations)
    splits_path = Path(getattr(cfg.paths, "split_manifest", "data/splits.json"))

    dataset = VieGameEmoDataset(
        annotations_dir=annotations_dir,
        features_dir=Path(cfg.paths.features),
        split=args.split,
        split_manifest=splits_path if splits_path.exists() else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.perception.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    all_preds, all_labels, all_meta = [], [], []
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            context = batch["context"].to(device)
            text = batch["text"].to(device)
            labels = batch["label"].to(device)
            has_face = batch.get("has_face")
            if has_face is not None:
                has_face = has_face.to(device)

            fused = fusion(audio, face, context, text, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]
            logits = classifier(fused)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            clip_ids = batch.get("clip_id", [""] * len(labels))
            genres = batch.get("genre", ["unknown"] * len(labels))
            for i in range(len(labels)):
                all_meta.append({
                    "clip_id": clip_ids[i] if isinstance(clip_ids, (list, tuple)) else "",
                    "genre": genres[i] if isinstance(genres, (list, tuple)) else "unknown",
                    "y_true": int(labels[i].item()),
                    "y_pred": int(preds[i].item()),
                })

    metrics = compute_metrics(
        all_labels, all_preds,
        n_classes=ccfg.n_classes,
        label_names=_EMOTION_LABELS,
    )
    cm = metrics.pop("confusion_matrix").tolist()
    metrics["confusion_matrix"] = cm

    result: dict = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "n_samples": len(all_labels),
        "metrics": metrics,
    }

    if args.per_genre:
        genre_m = per_genre_metrics(all_meta, genre_field="genre")
        result["genre_metrics"] = genre_m

    if args.include_reasoning:
        from vie_gameemo.evaluation.reasoning_eval import evaluate_reasoning
        reasoning_preds = [
            {
                "predicted_reasoning": "",
                "predicted_label": _EMOTION_LABELS[p] if p < len(_EMOTION_LABELS) else str(p),
                "gt_label": _EMOTION_LABELS[t] if t < len(_EMOTION_LABELS) else str(t),
                "gt_reasoning": "",
            }
            for p, t in zip(all_preds, all_labels)
        ]
        reasoning_result = evaluate_reasoning(
            reasoning_preds,
            judge_model=cfg.llm.base_model.name,
            quantization=cfg.llm.base_model.quantization,
        )
        reasoning_result.pop("per_sample", None)
        result["reasoning_eval"] = reasoning_result

    logger.info(
        "split=%s | acc=%.4f | macro_f1=%.4f | uar=%.4f",
        args.split,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["uar"],
    )
    return result


def _run_ablation(cfg, args) -> dict:
    """Run an ablation suite (strategy / fusion / llm)."""
    output_dir = Path(cfg.paths.checkpoints) / f"ablation_{args.ablation}"

    if args.ablation == "strategy":
        from vie_gameemo.evaluation.strategy_ablation import (
            format_ablation_table,
            run_strategy_ablation,
        )
        results = run_strategy_ablation(cfg, output_dir)
        logger.info("\n%s", format_ablation_table(results))
        return {"ablation": "strategy", "results": results}

    elif args.ablation == "fusion":
        return _run_fusion_ablation(cfg, output_dir)

    elif args.ablation == "llm":
        return _run_llm_ablation(cfg, args, output_dir)

    raise ValueError(f"Unknown ablation type: {args.ablation}")


def _run_fusion_ablation(cfg, output_dir: Path) -> dict:
    """Evaluate each registered fusion type on the same data split."""
    from vie_gameemo.fusion import _FUSION_REGISTRY, _ensure_registered

    _ensure_registered()
    fusion_types = list(_FUSION_REGISTRY.keys())
    logger.info("Fusion ablation: %s", fusion_types)

    results = {}
    for fusion_type in fusion_types:
        logger.info("--- Fusion: %s ---", fusion_type)
        import copy
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.fusion.type = fusion_type
        sub_cfg.paths.checkpoints = str(output_dir / f"fusion_{fusion_type}")

        from vie_gameemo.data.dataset import VieGameEmoDataset, collate_fn
        from vie_gameemo.evaluation.metrics import compute_metrics
        from vie_gameemo.training.perception import train_perception

        import torch
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        annotations_dir = Path(cfg.paths.annotations)
        splits_path = Path(getattr(cfg.paths, "split_manifest", "data/splits.json"))

        train_ds = VieGameEmoDataset(annotations_dir, Path(cfg.paths.features), "train",
                                     split_manifest=splits_path if splits_path.exists() else None)
        val_ds = VieGameEmoDataset(annotations_dir, Path(cfg.paths.features), "val",
                                   split_manifest=splits_path if splits_path.exists() else None)

        bs = sub_cfg.training.perception.batch_size
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

        try:
            ckpt = train_perception(sub_cfg, train_loader, val_loader, device)
            from vie_gameemo.classifiers.mlp import EmotionClassifier
            from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
            from vie_gameemo.training.perception import evaluate, load_checkpoint

            fcfg = sub_cfg.fusion
            ccfg = sub_cfg.classifier
            fusion_m = get_fusion(fusion_type, d_model=fcfg.d_model, n_modalities=fcfg.n_modalities,
                                  return_attention=False, **modality_dim_kwargs(fcfg)).to(device)
            cls_m = EmotionClassifier(fcfg.d_model, ccfg.hidden_dim, ccfg.n_classes, ccfg.dropout).to(device)
            load_checkpoint(ckpt, fusion_m, cls_m)
            test_ds = VieGameEmoDataset(annotations_dir, Path(cfg.paths.features), "test",
                                        split_manifest=splits_path if splits_path.exists() else None)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
            metrics = evaluate(fusion_m, cls_m, test_loader, device, ccfg.n_classes)
            metrics.pop("confusion_matrix", None)
            results[fusion_type] = metrics
        except Exception as exc:
            logger.warning("Fusion %s failed: %s", fusion_type, exc)
            results[fusion_type] = {"error": str(exc)}

    return {"ablation": "fusion", "results": results}


def _run_llm_ablation(cfg, args, output_dir: Path) -> dict:
    """Evaluate each LLM setup on val split using annotation reasoning data."""
    import json

    from vie_gameemo.evaluation.reasoning_eval import format_compliance

    llm_setups = {
        "llm1": "LLM1Explainer (soft token + MLP label, no training)",
        "llm2": "LLM2CoReasoner (soft token + MLP hint, may override)",
        "llm3": "LLM3PureReasoner (soft token only)",
        "llm4": "LLM4RLVR (RLVR-trained)",
    }

    annotations_dir = Path(cfg.paths.annotations)
    json_paths = sorted(annotations_dir.glob("*.json"))[:50]

    results = {}
    for setup_key, description in llm_setups.items():
        logger.info("--- LLM setup: %s ---", setup_key)
        llm = _instantiate_llm(cfg, setup_key)
        if llm is None:
            results[setup_key] = {"error": "not available"}
            continue

        preds = []
        for p in json_paths:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                evidence = {
                    "face_aus": data.get("face_aus", "N/A"),
                    "visual_objective": data.get("visual_objective", "N/A"),
                    "audio_tone": data.get("audio_tone", "N/A"),
                    "transcript": data.get("transcript", ""),
                    "label": data.get("emotion_label", "neutral"),
                }
                out = llm.reason(evidence)
                preds.append({
                    "predicted_label": out.answer,
                    "gt_label": data.get("emotion_label", "neutral"),
                    "predicted_reasoning": out.reasoning,
                    "gt_reasoning": data.get("reasoning", ""),
                    "raw": out.raw,
                })
            except Exception as exc:
                logger.warning("LLM %s failed on %s: %s", setup_key, p.name, exc)

        if hasattr(llm, "unload"):
            llm.unload()

        if preds:
            n_correct = sum(p["predicted_label"] == p["gt_label"] for p in preds)
            results[setup_key] = {
                "description": description,
                "n_samples": len(preds),
                "accuracy": n_correct / len(preds),
                "format_compliance": format_compliance(preds),
            }
        else:
            results[setup_key] = {"description": description, "error": "no predictions"}

    return {"ablation": "llm", "results": results}


def _instantiate_llm(cfg, setup_key: str):
    """Instantiate LLM setup by key, return None if not available."""
    try:
        if setup_key == "llm1":
            from vie_gameemo.llm.llm1_explainer import LLM1Explainer
            return LLM1Explainer(cfg.llm.base_model.name, quantization=cfg.llm.base_model.quantization)
        elif setup_key == "llm2":
            from vie_gameemo.llm.llm2_coreasoner import LLM2CoReasoner
            return LLM2CoReasoner(cfg.llm.base_model.name, quantization=cfg.llm.base_model.quantization)
        elif setup_key == "llm3":
            from vie_gameemo.llm.llm3_vlm import LLM3PureReasoner
            cognition_ckpt = getattr(cfg.llm, "cognition_checkpoint", None)
            return LLM3PureReasoner(
                model_name=cfg.llm.base_model.name,
                quantization=cfg.llm.base_model.quantization,
                modal_adapter_ckpt=Path(cognition_ckpt) if cognition_ckpt else None,
            )
        elif setup_key == "llm4":
            from vie_gameemo.llm.llm4_rlvr import LLM4RLVR
            return LLM4RLVR(cfg.llm.base_model.name, quantization=cfg.llm.base_model.quantization)
    except Exception as exc:
        logger.warning("Cannot instantiate %s: %s", setup_key, exc)
    return None


if __name__ == "__main__":
    sys.exit(main())
