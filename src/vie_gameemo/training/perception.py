"""Stage 1 — Perception trainer (classification only).

Trains fusion + classifier on emotion labels. Encoders are FROZEN (use cached
features from `extract_features.py`). LLM is NOT involved at this stage.

Output: checkpoint with trained fusion + classifier weights, ready for
Stage 2 — Cognition.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Mutable training state for checkpoint save/load and resume."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    patience_counter: int = 0


def train_perception(
    cfg: SimpleNamespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    resume_from: Path | None = None,
) -> Path:
    """Train Stage 1 — Perception (fusion + classifier).

    Args:
        cfg: Full config namespace (needs cfg.training.perception, cfg.fusion,
             cfg.classifier, cfg.paths.checkpoints).
        train_loader: DataLoader over cached features (train split).
        val_loader: DataLoader over cached features (val split).
        device: Torch device.
        resume_from: Optional checkpoint to resume from.

    Returns:
        Path to best checkpoint.
    """
    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
    from vie_gameemo.training.losses import FocalLoss, make_class_weights

    pcfg = cfg.training.perception
    fcfg = cfg.fusion
    ccfg = cfg.classifier

    # Build model: fusion + classifier
    fusion = get_fusion(
        fcfg.type,
        d_model=fcfg.d_model,
        n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
        **modality_dim_kwargs(fcfg),
    ).to(device)

    classifier = EmotionClassifier(
        d_model=fcfg.d_model,
        hidden_dim=ccfg.hidden_dim,
        n_classes=ccfg.n_classes,
        dropout=ccfg.dropout,
    ).to(device)

    # Class weights for focal/weighted_ce
    loss_cfg = ccfg.loss
    loss_type = getattr(loss_cfg, "type", "focal")
    class_weight_method = getattr(loss_cfg, "class_weights", "none")
    alpha = getattr(loss_cfg.focal, "alpha", 1.0)

    if class_weight_method != "none":
        all_train_labels = [item["label"] for item in train_loader.dataset.items]
        alpha = make_class_weights(all_train_labels, ccfg.n_classes, method=class_weight_method)
        logger.info("Using %s class weights: %s", class_weight_method, alpha)

    criterion = FocalLoss(
        gamma=getattr(loss_cfg.focal, "gamma", 2.0),
        alpha=alpha,
    )

    params = list(fusion.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=pcfg.learning_rate.fusion,
        weight_decay=pcfg.weight_decay,
    )

    # Cosine scheduler with linear warmup
    n_steps = len(train_loader) * pcfg.epochs
    warmup_steps = int(n_steps * pcfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    state = TrainingState()
    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "perception_best.pt"

    if resume_from and Path(resume_from).exists():
        state = load_checkpoint(resume_from, fusion, classifier, optimizer)
        logger.info("Resumed from %s at epoch %d", resume_from, state.epoch)

    use_amp = getattr(pcfg, "mixed_precision", "no") in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if getattr(pcfg, "mixed_precision", "no") == "bf16" else torch.float16
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    grad_accum = getattr(pcfg, "gradient_accumulation", 1)
    grad_clip = getattr(pcfg, "grad_clip", 1.0)

    early_stopping = getattr(pcfg, "early_stopping", None)
    patience = getattr(early_stopping, "patience", 5) if early_stopping else 999

    logger.info("Perception training: %d epochs, %d steps/epoch", pcfg.epochs, len(train_loader))

    for epoch in range(state.epoch, pcfg.epochs):
        state.epoch = epoch
        fusion.train()
        classifier.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            context = batch["context"].to(device)
            text = batch["text"].to(device)
            labels = batch["label"].to(device)
            has_face = batch.get("has_face")
            if has_face is not None:
                has_face = has_face.to(device)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                fused = fusion(audio, face, context, text, has_face=has_face)
                if isinstance(fused, tuple):
                    fused = fused[0]
                logits = classifier(fused)
                loss = criterion(logits, labels) / grad_accum

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                if grad_clip > 0:
                    if use_amp and amp_dtype == torch.float16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                if use_amp and amp_dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                state.global_step += 1

        avg_loss = train_loss / len(train_loader)
        val_metrics = evaluate(
            fusion=fusion, classifier=classifier, loader=val_loader,
            device=device, n_classes=ccfg.n_classes,
            return_predictions=True,
        )
        macro_f1 = val_metrics["macro_f1"]
        logger.info(
            "Epoch %d/%d | loss=%.4f | val_macro_f1=%.4f | val_uar=%.4f",
            epoch + 1, pcfg.epochs, avg_loss, macro_f1, val_metrics["uar"],
        )
        if "per_class_f1" in val_metrics:
            for cls_name, f1_val in val_metrics["per_class_f1"].items():
                logger.info("  %s: F1=%.4f recall=%.4f",
                            cls_name, f1_val, val_metrics["per_class_recall"].get(cls_name, 0))

        if macro_f1 > state.best_metric:
            state.best_metric = macro_f1
            state.patience_counter = 0
            _save_checkpoint(best_ckpt, fusion, classifier, optimizer, state)
            logger.info("New best model saved (macro_f1=%.4f)", macro_f1)

            # Save misclassified clips for dataset review
            predictions = val_metrics.get("predictions", [])
            errors = [p for p in predictions if not p["correct"]]
            if errors:
                import json
                errors_path = ckpt_dir / "misclassified_best_epoch.json"
                errors_path.write_text(
                    json.dumps(errors, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("Saved %d misclassified clips → %s", len(errors), errors_path)
        else:
            state.patience_counter += 1
            if state.patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    logger.info("Perception training done. Best macro_f1=%.4f", state.best_metric)
    return best_ckpt


def evaluate(
    fusion: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    return_predictions: bool = False,
) -> dict:
    """Evaluate fusion+classifier on a loader.

    Args:
        fusion: Fusion module.
        classifier: Classifier module.
        loader: DataLoader.
        device: Torch device.
        n_classes: Number of emotion classes.
        return_predictions: If True, include per-sample predictions in output.

    Returns:
        Dict with keys: 'accuracy', 'macro_f1', 'weighted_f1', 'uar'.
        If return_predictions: also 'predictions' list of {clip_id, gt, pred, correct}.
    """
    fusion.eval()
    classifier.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_clip_ids: list[str] = []

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
            if isinstance(clip_ids, (list, tuple)):
                all_clip_ids.extend(clip_ids)
            else:
                all_clip_ids.extend([""] * len(labels))

    from vie_gameemo.training.losses import per_class_metrics as compute_per_class
    from vie_gameemo.data.schemas import EmotionLabel

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(1, len(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    from sklearn.metrics import recall_score
    uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    class_names = [e.value for e in EmotionLabel]
    detailed = compute_per_class(all_preds, all_labels, n_classes, class_names)

    result = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "uar": float(uar),
        "per_class_f1": detailed["per_class_f1"],
        "per_class_recall": detailed["per_class_recall"],
        "per_class_precision": detailed["per_class_precision"],
        "confusion_matrix": detailed["confusion_matrix"],
    }

    if return_predictions:
        predictions = []
        for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
            gt_name = class_names[label] if label < len(class_names) else str(label)
            pred_name = class_names[pred] if pred < len(class_names) else str(pred)
            predictions.append({
                "clip_id": all_clip_ids[i] if i < len(all_clip_ids) else "",
                "gt": gt_name,
                "pred": pred_name,
                "correct": pred == label,
            })
        result["predictions"] = predictions

    return result


def _save_checkpoint(
    path: Path,
    fusion: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
) -> None:
    """Atomically save training checkpoint.

    Args:
        path: Output path.
        fusion: Fusion module.
        classifier: Classifier module.
        optimizer: Optimizer state.
        state: Training state.
    """
    tmp = path.with_suffix(".tmp")
    torch.save({
        "fusion_state_dict": fusion.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
    }, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    fusion: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainingState:
    """Load checkpoint and restore module states.

    Args:
        path: Checkpoint path.
        fusion: Fusion module (mutated in place).
        classifier: Classifier module (mutated in place).
        optimizer: Optional optimizer (mutated in place).

    Returns:
        Restored TrainingState.
    """
    ckpt = torch.load(path, map_location="cpu")
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return TrainingState(
        epoch=ckpt.get("epoch", 0),
        global_step=ckpt.get("global_step", 0),
        best_metric=ckpt.get("best_metric", float("-inf")),
    )


def infer_fusion_dims_from_checkpoint(path: Path, d_model: int) -> dict[str, int]:
    """Recover per-modality encoder dims from a saved fusion checkpoint.

    The Conv-Attention fusion's per-modality MLPs (``mlp_{audio,face,context,
    text}.weight``) have shape ``(d_model, modality_dim)``, so reading back
    ``shape[1]`` tells us the dim the checkpoint was trained with. This lets
    eval rebuild the exact architecture independent of config drift (e.g. a
    checkpoint trained with text_dim=1024 for XLM-R-large/CafeBERT evaluated
    against a config that forgot to set it).

    Returns only dims that differ from ``d_model`` — the standard 768-dim
    modalities need no override, and emitting them would break baseline
    fusions that don't accept ``*_dim`` kwargs.
    """
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("fusion_state_dict", {})
    dims: dict[str, int] = {}
    for mod in ("audio", "face", "context", "text"):
        w = sd.get(f"mlp_{mod}.weight")
        if w is not None and w.shape[1] != d_model:
            dims[f"{mod}_dim"] = int(w.shape[1])
    return dims
