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
    parser.add_argument(
        "--fusion", type=str, default=None,
        help="Override fusion type (for checkpoints predating the saved "
             "fusion_type field, or to force a specific architecture)",
    )
    parser.add_argument(
        "--zero-modality", action="append", choices=["audio", "face", "context", "text"],
        default=None, dest="zero_modality",
        help=(
            "Ablation: zero out a modality at eval time (repeatable). Should "
            "match whatever --zero-modality was used at train time to "
            "measure that model correctly; use a different value only to "
            "deliberately test robustness to a missing modality."
        ),
    )
    parser.add_argument(
        "--llm-perception-ckpt", type=Path, default=None,
        help=(
            "Evaluate a Stage 2a 'llm_perception_best.pt' checkpoint (from "
            "train.py --stage llm_perception) instead of the MLP perception "
            "checkpoint. Reuses the same in-training eval "
            "(cognition._eval_llm_metrics) on a full split with a "
            "configurable --n-samples, rather than training's fixed "
            "50-sample per-epoch check. If the checkpoint was trained with "
            "use_mlp_hint=True, also pass --checkpoint pointing to the "
            "matching perception_best.pt (needed to load the frozen "
            "classifier for the hint)."
        ),
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Max samples to evaluate for --llm-perception-ckpt (generation "
             "is slow — this is a subset of --split, not the whole split).",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/results/eval.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cli_overrides = {"fusion.type": args.fusion} if args.fusion is not None else None
    cfg = load_config(args.config, cli_overrides=cli_overrides)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)

    if args.llm_perception_ckpt:
        results = _run_llm_perception_eval(cfg, args)
    elif args.ablation:
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

    from vie_gameemo.classifiers import get_classifier
    from vie_gameemo.data.dataset import VieGameEmoDataset, collate_fn, zero_modalities_for_strategy
    from vie_gameemo.data.schemas import resolve_labels
    from vie_gameemo.evaluation.metrics import compute_metrics
    from vie_gameemo.evaluation.per_genre import per_genre_metrics
    from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
    from vie_gameemo.training.perception import (
        infer_classifier_type_from_checkpoint,
        infer_fusion_dims_from_checkpoint,
        infer_fusion_type_from_checkpoint,
        load_checkpoint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fcfg = cfg.fusion
    ccfg = cfg.classifier

    # Build fusion to match the checkpoint's actual architecture. The checkpoint
    # is the source of truth for both fusion type and per-modality dims
    # (text_dim, etc.); fall back to config only when the checkpoint predates
    # these fields (trained before this was saved).
    ckpt_fusion_type = infer_fusion_type_from_checkpoint(args.checkpoint)
    fusion_type = ckpt_fusion_type or fcfg.type
    if ckpt_fusion_type and ckpt_fusion_type != fcfg.type:
        logger.warning(
            "Checkpoint was trained with fusion='%s' but config.yaml has "
            "fusion.type='%s'; using the checkpoint's type.",
            ckpt_fusion_type, fcfg.type,
        )

    dim_kwargs = modality_dim_kwargs(fcfg)
    dim_kwargs.update(infer_fusion_dims_from_checkpoint(args.checkpoint, fcfg.d_model))
    fusion = get_fusion(
        fusion_type,
        d_model=fcfg.d_model,
        n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
        skip_mlp_if_matched=getattr(fcfg, "skip_mlp_if_matched", False),
        **dim_kwargs,
    ).to(device)
    ckpt_classifier_type = infer_classifier_type_from_checkpoint(args.checkpoint)
    classifier = get_classifier(ccfg, d_model=fcfg.d_model, device=device, classifier_type=ckpt_classifier_type)
    load_checkpoint(args.checkpoint, fusion, classifier)
    fusion.eval()
    classifier.eval()

    annotations_dir = Path(cfg.paths.annotations)
    splits_path = Path(getattr(cfg.paths, "split_manifest", "data/splits.json"))

    strategy = getattr(getattr(cfg, "visual_encoder", None), "strategy", "dual_path")
    zero_mods = sorted(set(zero_modalities_for_strategy(strategy)) | set(args.zero_modality or []))
    if zero_mods:
        logger.info(
            "Zeroing modalities at load time: %s (strategy='%s' + --zero-modality=%s)",
            zero_mods, strategy, args.zero_modality or [],
        )

    dataset = VieGameEmoDataset(
        annotations_dir=annotations_dir,
        features_dir=Path(cfg.paths.features),
        split=args.split,
        merge_mode=getattr(cfg.labeling, "merge_mode", "none"),
        split_manifest=splits_path if splits_path.exists() else None,
        zero_modalities=zero_mods,
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

    class_names = resolve_labels(getattr(cfg.labeling, "merge_mode", "none"))[0]
    metrics = compute_metrics(
        all_labels, all_preds,
        n_classes=ccfg.n_classes,
        label_names=class_names,
    )
    cm = metrics.pop("confusion_matrix").tolist()
    metrics["confusion_matrix"] = cm

    # Per-sample predictions (for misclassification analysis)
    predictions = []
    for m in all_meta:
        predictions.append({
            "clip_id": m["clip_id"],
            "gt": class_names[m["y_true"]] if m["y_true"] < len(class_names) else str(m["y_true"]),
            "pred": class_names[m["y_pred"]] if m["y_pred"] < len(class_names) else str(m["y_pred"]),
            "correct": m["y_true"] == m["y_pred"],
        })

    result: dict = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "n_samples": len(all_labels),
        "metrics": metrics,
        "predictions": predictions,
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


def _run_llm_perception_eval(cfg, args) -> dict:
    """Standalone eval of a Stage 2a 'llm_perception_best.pt' checkpoint.

    Reuses `cognition._eval_llm_metrics` (the same metric computed during
    training's per-epoch check) but on a full split with a configurable
    sample budget, instead of training's fixed 50-sample check.
    """
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.data.dataset import VieGameEmoDataset, collate_fn
    from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
    from vie_gameemo.llm.modal_adapter import ModalAdapter
    from vie_gameemo.training.cognition import _eval_llm_metrics, _make_bnb_config
    from vie_gameemo.utils.torch_compat import ensure_set_submodule_patch

    ensure_set_submodule_patch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    ckpt = torch.load(args.llm_perception_ckpt, map_location="cpu", weights_only=False)
    use_mlp_hint = ckpt.get("use_mlp_hint", False)

    # Fusion: this checkpoint's own fine-tuned copy (NOT perception_best.pt's).
    fusion = get_fusion(
        fcfg.type, d_model=fcfg.d_model, n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
        skip_mlp_if_matched=getattr(fcfg, "skip_mlp_if_matched", False),
        **modality_dim_kwargs(fcfg, features_dir=Path(cfg.paths.features)),
    ).to(device)
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    fusion.eval()

    # Classifier only needed for the MLP-hint prompt variant; llm_perception_best.pt
    # doesn't save it (classifier stays frozen), so it must come from perception_best.pt.
    classifier = None
    if use_mlp_hint:
        if not args.checkpoint:
            raise ValueError(
                "This llm_perception checkpoint was trained with use_mlp_hint=True — "
                "pass --checkpoint pointing to the matching perception_best.pt "
                "to load the classifier needed for the hint."
            )
        from vie_gameemo.classifiers import get_classifier
        from vie_gameemo.training.perception import infer_classifier_type_from_checkpoint

        p_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        classifier = get_classifier(
            ccfg, d_model=fcfg.d_model, device=device,
            classifier_type=infer_classifier_type_from_checkpoint(args.checkpoint),
        )
        classifier.load_state_dict(p_ckpt["classifier_state_dict"])
        classifier.eval()

    model_name = llm_cfg.base_model.name
    quant_cfg = _make_bnb_config(llm_cfg.base_model.quantization)
    lm_kwargs: dict = {"device_map": "auto"}
    if quant_cfg is not None:
        lm_kwargs["quantization_config"] = quant_cfg
    else:
        lm_kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name, **lm_kwargs)

    if ckpt.get("llm_peft") is not None:
        from peft import LoraConfig, get_peft_model

        lp_cfg = getattr(cfg.training, "llm_perception", None) or cfg.training.cognition
        lora_cfg_ns = lp_cfg.lora
        lora_config = LoraConfig(
            r=lora_cfg_ns.rank, lora_alpha=lora_cfg_ns.alpha,
            target_modules=list(lora_cfg_ns.target_modules),
            bias="none", task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        llm.load_state_dict(ckpt["llm_peft"], strict=False)
    llm.eval()

    llm_hidden_size = llm.config.hidden_size
    llm_adapter = ModalAdapter(d_fusion=fcfg.d_model, d_llm=llm_hidden_size).to(device)
    llm_adapter.load_state_dict(ckpt["llm_adapter"])
    llm_adapter.eval()

    annotations_dir = Path(cfg.paths.annotations)
    splits_path = Path(getattr(cfg.paths, "split_manifest", "data/splits.json"))
    dataset = VieGameEmoDataset(
        annotations_dir=annotations_dir,
        features_dir=Path(cfg.paths.features),
        split=args.split,
        merge_mode=getattr(cfg.labeling, "merge_mode", "none"),
        split_manifest=splits_path if splits_path.exists() else None,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    metrics = _eval_llm_metrics(
        fusion, llm_adapter, llm, tokenizer, loader, device,
        use_mlp_hint, classifier, n_samples=args.n_samples,
    )

    logger.info(
        "llm_perception eval | split=%s | acc=%.4f | macro_f1=%.4f | format_rate=%.4f | n=%d",
        args.split, metrics["accuracy"], metrics["macro_f1"],
        metrics["format_rate"], metrics["n_samples"],
    )
    return {
        "stage": "llm_perception",
        "checkpoint": str(args.llm_perception_ckpt),
        "use_mlp_hint": use_mlp_hint,
        "split": args.split,
        "metrics": metrics,
    }


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

        merge_mode = getattr(cfg.labeling, "merge_mode", "none")
        train_ds = VieGameEmoDataset(annotations_dir, Path(cfg.paths.features), "train",
                                     merge_mode=merge_mode,
                                     split_manifest=splits_path if splits_path.exists() else None)
        val_ds = VieGameEmoDataset(annotations_dir, Path(cfg.paths.features), "val",
                                   merge_mode=merge_mode,
                                   split_manifest=splits_path if splits_path.exists() else None)

        bs = sub_cfg.training.perception.batch_size
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

        try:
            ckpt = train_perception(sub_cfg, train_loader, val_loader, device)
            from vie_gameemo.classifiers import get_classifier
            from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
            from vie_gameemo.training.perception import evaluate, load_checkpoint

            fcfg = sub_cfg.fusion
            ccfg = sub_cfg.classifier
            fusion_m = get_fusion(fusion_type, d_model=fcfg.d_model, n_modalities=fcfg.n_modalities,
                                  return_attention=False,
                                  skip_mlp_if_matched=getattr(fcfg, "skip_mlp_if_matched", False),
                                  **modality_dim_kwargs(fcfg)).to(device)
            cls_m = get_classifier(ccfg, d_model=fcfg.d_model, device=device)
            load_checkpoint(ckpt, fusion_m, cls_m)
            test_ds = VieGameEmoDataset(annotations_dir, Path(cfg.paths.features), "test",
                                        merge_mode=merge_mode,
                                        split_manifest=splits_path if splits_path.exists() else None)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
            from vie_gameemo.data.schemas import resolve_labels
            metrics = evaluate(fusion_m, cls_m, test_loader, device, ccfg.n_classes,
                              class_names=resolve_labels(merge_mode)[0])
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
