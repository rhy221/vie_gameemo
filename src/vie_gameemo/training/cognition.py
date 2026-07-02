"""LLM training stages.

Two stages for LLM, run after MLP perception (Stage 1):

Stage 2a — LLM Perception (requires only GT labels):
    Train ModalAdapter + LLM LoRA to predict emotion labels from soft tokens.
    Input: [soft_token | instruction + label_choices] → target: <answer>{gt_label}</answer>
    No annotated descriptions needed.

Stage 2b — Cognition (optional, requires annotated descriptions):
    Joint recognition + reasoning instruction tuning.
    Loss = α * L_classification + β * L_reasoning_LM
"""

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


_LABEL_NAMES = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


def _maybe_balanced_sampler(train_loader: DataLoader, cfg) -> DataLoader:
    """Rebuild train_loader with WeightedRandomSampler if config requests it."""
    from torch.utils.data import WeightedRandomSampler

    sampler_type = getattr(getattr(cfg, "classifier", None), "sampler", "none")
    if sampler_type != "balanced_batch":
        return train_loader

    n_classes = getattr(cfg.classifier, "n_classes", len(_LABEL_NAMES))
    all_labels = [item["label"] for item in train_loader.dataset.items]
    class_counts = torch.zeros(n_classes)
    for lbl in all_labels:
        class_counts[lbl] += 1
    sample_weights = 1.0 / class_counts.clamp(min=1.0)
    per_sample_weight = [float(sample_weights[lbl]) for lbl in all_labels]

    sampler = WeightedRandomSampler(
        per_sample_weight, num_samples=len(per_sample_weight), replacement=True,
    )
    logger.info("Using balanced_batch sampler for LLM training (oversampling rare classes)")
    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        collate_fn=train_loader.collate_fn,
    )

_LLM_PERCEPTION_PROMPT = (
    "Dựa trên đặc trưng đa phương thức của clip game livestream, "
    "hãy xác định cảm xúc của streamer.\n"
    "Các nhãn có thể: {labels}\n"
    "Trả lời theo format: <answer>[nhãn]</answer>"
)

_LLM_PERCEPTION_PROMPT_WITH_HINT = (
    "Classifier gợi ý streamer đang ở trạng thái: {mlp_label}. "
    "Đây chỉ là gợi ý — có thể đúng hoặc sai.\n"
    "Dựa trên đặc trưng đa phương thức, hãy tự xác định cảm xúc.\n"
    "Các nhãn có thể: {labels}\n"
    "Trả lời theo format: <answer>[nhãn]</answer>"
)

# --- Multi-task prompts for cognition training ---

_TASK_AUDIO_PROMPT = (
    "Dựa trên đặc trưng âm thanh của clip game livestream, hãy phân tích "
    "giọng nói và âm thanh của streamer. Mô tả ngữ điệu, tốc độ nói, "
    "cường độ, và các đặc điểm cảm xúc trong giọng nói."
)

_TASK_VISUAL_PROMPT = (
    "Dựa trên đặc trưng hình ảnh của clip game livestream, hãy mô tả "
    "biểu cảm khuôn mặt, cử chỉ, và ngôn ngữ cơ thể của streamer."
)

_TASK_REASONING_PROMPT = (
    "Dựa trên đặc trưng đa phương thức của clip game livestream, "
    "streamer đang ở trạng thái {label}.\n"
    "Hãy giải thích vì sao, liên kết bằng chứng từ nhiều modality.\n"
    "Trả lời theo format:\n"
    "<think>[lý luận]</think>\n<answer>{label}</answer>"
)

_COGNITION_TASKS = ["emotion", "audio", "visual", "reasoning"]


def train_llm_perception(
    cfg: SimpleNamespace,
    perception_checkpoint: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    use_mlp_hint: bool = False,
) -> Path:
    """Stage 2a — LLM Perception: align soft tokens to predict emotion labels.

    Only requires ground-truth labels, no annotated descriptions.
    Trains ModalAdapter + LLM LoRA to generate <answer>{label}</answer>
    from fusion embedding soft tokens.

    Args:
        cfg: Full config namespace.
        perception_checkpoint: Stage 1 checkpoint (fusion + classifier).
        train_loader: DataLoader with cached features + labels.
        val_loader: Validation DataLoader.
        device: Torch device.
        use_mlp_hint: If True, include MLP prediction as hint in prompt (LLM-2 style).

    Returns:
        Path to best checkpoint.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.fusion import get_fusion

    lp_cfg = getattr(cfg.training, "llm_perception", None) or cfg.training.cognition
    pcfg = cfg.training.perception
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # Load fusion (init from MLP perception, fine-tune for LLM) + classifier (frozen)
    fusion = get_fusion(
        fcfg.type, d_model=fcfg.d_model, n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
    ).to(device)
    classifier = EmotionClassifier(
        d_model=fcfg.d_model, hidden_dim=ccfg.hidden_dim,
        n_classes=ccfg.n_classes, dropout=ccfg.dropout,
    ).to(device)

    import sys, types
    sys.modules.setdefault("torch.utils.serialization", types.ModuleType("torch.utils.serialization"))
    ckpt = torch.load(perception_checkpoint, map_location="cpu", weights_only=False)
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    # Classifier always frozen — MLP prediction stays unchanged
    for p in classifier.parameters():
        p.requires_grad = False
    classifier.eval()

    # Fusion: fine-tune a separate copy for LLM (init from MLP perception weights)
    # Original MLP fusion is preserved in perception_best.pt
    fusion.train()
    fusion_lr = getattr(lp_cfg.learning_rate, "fusion", 2e-5)
    logger.info("Fusion: trainable for LLM (lr=%.1e), init from %s", fusion_lr, perception_checkpoint)

    # Load LLM with LoRA
    model_name = llm_cfg.base_model.name
    logger.info("Loading LLM: %s", model_name)
    quant_cfg = _make_bnb_config(llm_cfg.base_model.quantization)
    lm_kwargs: dict = {"device_map": "auto"}
    if quant_cfg is not None:
        lm_kwargs["quantization_config"] = quant_cfg
    else:
        lm_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name, **lm_kwargs)

    lora_cfg_ns = lp_cfg.lora
    if getattr(lora_cfg_ns, "enabled", True):
        lora_config = LoraConfig(
            r=lora_cfg_ns.rank, lora_alpha=lora_cfg_ns.alpha,
            target_modules=list(lora_cfg_ns.target_modules),
            bias="none", task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()

    from vie_gameemo.llm.modal_adapter import ModalAdapter
    llm_hidden_size = llm.config.hidden_size
    llm_adapter = ModalAdapter(d_fusion=fcfg.d_model, d_llm=llm_hidden_size).to(device)

    trainable_params = [
        {"params": fusion.parameters(), "lr": fusion_lr},
        {"params": llm_adapter.parameters()},
        {"params": llm.parameters()},
    ]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lp_cfg.learning_rate.llm,
        weight_decay=getattr(pcfg, "weight_decay", 0.01),
    )

    # Balanced sampler for LLM perception (same imbalance problem)
    train_loader = _maybe_balanced_sampler(train_loader, cfg)

    n_epochs = lp_cfg.epochs
    n_steps = len(train_loader) * n_epochs
    warmup_steps = max(1, int(n_steps * getattr(pcfg, "warmup_ratio", 0.1)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "llm_perception_best.pt"

    labels_str = ", ".join(_LABEL_NAMES)
    mode_name = "LLM Perception (with MLP hint)" if use_mlp_hint else "LLM Perception"
    logger.info("%s training: %d epochs", mode_name, n_epochs)
    best_metric = float("-inf")
    global_step = 0

    for epoch in range(n_epochs):
        fusion.train()
        llm.train()
        llm_adapter.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            context = batch["context"].to(device)
            text_feat = batch["text"].to(device)
            gt_labels = batch["label"].to(device)
            has_face = batch.get("has_face")
            if has_face is not None:
                has_face = has_face.to(device)

            B = audio.shape[0]

            # Fusion is trainable — gradient flows back to fine-tune for LLM
            fused = fusion(audio, face, context, text_feat, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]

            soft_tokens, soft_mask = llm_adapter(
                fused, audio=audio, face=face, context=context,
                has_face=has_face,
            )

            # Build per-sample prompts + targets
            prompts = []
            targets = []
            for i in range(B):
                gt_name = _LABEL_NAMES[gt_labels[i].item()]
                if use_mlp_hint:
                    with torch.no_grad():
                        mlp_logits = classifier(fused[i:i+1])
                        mlp_idx = int(mlp_logits.argmax(dim=-1).item())
                    mlp_name = _LABEL_NAMES[mlp_idx]
                    prompt = _LLM_PERCEPTION_PROMPT_WITH_HINT.format(
                        mlp_label=mlp_name, labels=labels_str,
                    )
                else:
                    prompt = _LLM_PERCEPTION_PROMPT.format(labels=labels_str)

                target = f"<think>\n</think>\n<answer>{gt_name}</answer>"
                prompts.append(prompt)
                targets.append(target)

            # Tokenize prompt + target together for causal LM loss
            full_texts = [p + "\n" + t for p, t in zip(prompts, targets)]
            prompt_only = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=256,
            ).to(device)
            full = tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True, max_length=300,
            ).to(device)

            embed_fn = llm.get_input_embeddings()
            full_embeds = embed_fn(full["input_ids"])  # (B, L, H)

            # Prepend soft tokens (multi-stream: fusion + per-modality)
            n_soft = soft_tokens.shape[1]
            inputs_embeds = torch.cat([soft_tokens, full_embeds], dim=1)
            attn_mask = torch.cat([soft_mask, full["attention_mask"]], dim=1)

            # Labels: mask soft tokens + prompt tokens, only compute loss on target tokens
            lm_labels = full["input_ids"].clone()
            for i in range(B):
                prompt_len = prompt_only["attention_mask"][i].sum().item()
                lm_labels[i, :prompt_len] = -100
            lm_labels = torch.cat([
                torch.full((B, n_soft), -100, device=device, dtype=torch.long),
                lm_labels,
            ], dim=1)

            lm_out = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=lm_labels,
            )

            loss = lm_out.loss
            loss.backward()
            epoch_loss += loss.item()

            grad_accum = getattr(lp_cfg, "gradient_accumulation", 1)
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            global_step += 1

        avg_loss = epoch_loss / max(1, len(train_loader))

        llm_metrics = _eval_llm_metrics(
            fusion, llm_adapter, llm, tokenizer, val_loader,
            device, use_mlp_hint, classifier, n_samples=50,
        )

        logger.info(
            "Epoch %d/%d | loss=%.4f | llm_macro_f1=%.4f | llm_uar=%.4f | format=%.2f",
            epoch + 1, n_epochs, avg_loss,
            llm_metrics["macro_f1"], llm_metrics["uar"], llm_metrics["format_rate"],
        )
        if llm_metrics["per_class_f1"]:
            for cls_name, f1_val in llm_metrics["per_class_f1"].items():
                logger.info("  %s: F1=%.4f recall=%.4f prec=%.4f",
                            cls_name, f1_val,
                            llm_metrics["per_class_recall"].get(cls_name, 0),
                            llm_metrics["per_class_precision"].get(cls_name, 0))

        llm_f1 = llm_metrics["macro_f1"]
        if llm_f1 > best_metric:
            best_metric = llm_f1
            torch.save({
                "fusion_state_dict": fusion.state_dict(),
                "llm_adapter": llm_adapter.state_dict(),
                "llm_peft": llm.state_dict() if hasattr(llm, "peft_config") else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "llm_metrics": llm_metrics,
                "use_mlp_hint": use_mlp_hint,
            }, best_ckpt)
            logger.info("New best LLM perception model saved (macro_f1=%.4f)", llm_f1)

    logger.info("%s training done. Best llm_macro_f1=%.4f", mode_name, best_metric)
    return best_ckpt


def _eval_llm_metrics(
    fusion, llm_adapter, llm, tokenizer, val_loader,
    device, use_mlp_hint, classifier, n_samples=50,
) -> dict:
    """Evaluate LLM's own label predictions on val set.

    Returns accuracy, macro_f1, per-class f1, and format compliance.
    """
    import re
    from collections import Counter

    fusion.eval()
    llm.eval()
    llm_adapter.eval()

    labels_str = ", ".join(_LABEL_NAMES)
    gt_list = []
    pred_list = []
    n_valid_format = 0
    total = 0

    for batch in val_loader:
        if total >= n_samples:
            break

        audio = batch["audio"].to(device)
        face = batch["face"].to(device)
        context = batch["context"].to(device)
        text_feat = batch["text"].to(device)
        gt_labels = batch["label"].to(device)
        has_face = batch.get("has_face")
        if has_face is not None:
            has_face = has_face.to(device)

        B = audio.shape[0]

        with torch.no_grad():
            fused = fusion(audio, face, context, text_feat, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]
            soft_tokens, _ = llm_adapter(
                fused, audio=audio, face=face, context=context,
                has_face=has_face,
            )

            for i in range(min(B, n_samples - total)):
                gt_idx = gt_labels[i].item()
                gt_name = _LABEL_NAMES[gt_idx]

                if use_mlp_hint:
                    mlp_logits = classifier(fused[i:i+1])
                    mlp_name = _LABEL_NAMES[int(mlp_logits.argmax(dim=-1).item())]
                    prompt = _LLM_PERCEPTION_PROMPT_WITH_HINT.format(
                        mlp_label=mlp_name, labels=labels_str,
                    )
                else:
                    prompt = _LLM_PERCEPTION_PROMPT.format(labels=labels_str)

                text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                text_embeds = llm.get_input_embeddings()(text_ids)
                inputs_embeds = torch.cat([soft_tokens[i:i+1], text_embeds], dim=1)

                out_ids = llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                raw = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

                match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
                if match:
                    n_valid_format += 1
                    predicted = match.group(1).strip().lower()
                else:
                    predicted = ""

                pred_idx = _LABEL_NAMES.index(predicted) if predicted in _LABEL_NAMES else -1
                gt_list.append(gt_idx)
                pred_list.append(pred_idx)
                total += 1

    fusion.train()
    llm.train()
    llm_adapter.train()

    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, precision_score,
    )

    # Map invalid predictions to -1 for sklearn
    valid_mask = [p >= 0 for p in pred_list]
    gt_arr = [gt_list[i] for i in range(len(gt_list)) if valid_mask[i]]
    pred_arr = [pred_list[i] for i in range(len(pred_list)) if valid_mask[i]]

    if not gt_arr:
        return {
            "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0,
            "uar": 0.0, "format_rate": 0.0, "n_samples": total,
            "per_class_f1": {}, "per_class_recall": {}, "per_class_precision": {},
        }

    accuracy = accuracy_score(gt_arr, pred_arr)
    macro_f1 = f1_score(gt_arr, pred_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(gt_arr, pred_arr, average="weighted", zero_division=0)
    uar = recall_score(gt_arr, pred_arr, average="macro", zero_division=0)
    format_rate = n_valid_format / max(1, total)

    # Per-class metrics
    n_classes = len(_LABEL_NAMES)
    per_f1 = f1_score(gt_arr, pred_arr, average=None, labels=range(n_classes), zero_division=0)
    per_recall = recall_score(gt_arr, pred_arr, average=None, labels=range(n_classes), zero_division=0)
    per_precision = precision_score(gt_arr, pred_arr, average=None, labels=range(n_classes), zero_division=0)

    gt_classes = set(gt_arr)
    per_class_f1 = {_LABEL_NAMES[c]: float(per_f1[c]) for c in gt_classes}
    per_class_recall = {_LABEL_NAMES[c]: float(per_recall[c]) for c in gt_classes}
    per_class_precision = {_LABEL_NAMES[c]: float(per_precision[c]) for c in gt_classes}

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "uar": float(uar),
        "format_rate": float(format_rate),
        "n_samples": total,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
    }


def train_cognition(
    cfg: SimpleNamespace,
    perception_checkpoint: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    llm_perception_checkpoint: Path | None = None,
) -> Path:
    """Train Stage 2b — Cognition (joint cls + reasoning).

    Continues from Stage 2a (LLM Perception) if checkpoint provided,
    otherwise starts fresh from Stage 1 (MLP Perception).

    Args:
        cfg: Full config namespace.
        perception_checkpoint: Stage 1 checkpoint (fusion + classifier).
        train_loader: DataLoader with batches including reasoning targets.
        val_loader: Validation DataLoader.
        device: Torch device.
        llm_perception_checkpoint: Optional Stage 2a checkpoint to continue
            from (loads fusion_v2 + ModalAdapter + LLM LoRA).

    Returns:
        Path to best checkpoint.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.fusion import get_fusion
    from vie_gameemo.training.losses import FocalLoss

    ccfg_train = cfg.training.cognition
    pcfg = cfg.training.perception
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # Determine which checkpoint to load fusion from:
    # - If Stage 2a checkpoint exists, use fusion_v2 (optimized for LLM)
    # - Otherwise fall back to fusion_v1 from Stage 1
    fusion_source = llm_perception_checkpoint or perception_checkpoint

    fusion = get_fusion(
        fcfg.type,
        d_model=fcfg.d_model,
        n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
    ).to(device)
    classifier = EmotionClassifier(
        d_model=fcfg.d_model,
        hidden_dim=ccfg.hidden_dim,
        n_classes=ccfg.n_classes,
        dropout=ccfg.dropout,
    ).to(device)

    import sys, types
    sys.modules.setdefault("torch.utils.serialization", types.ModuleType("torch.utils.serialization"))
    fusion_ckpt = torch.load(fusion_source, map_location="cpu", weights_only=False)
    fusion.load_state_dict(fusion_ckpt["fusion_state_dict"])
    logger.info("Loaded fusion from %s", fusion_source)

    # Classifier always from Stage 1 perception
    cls_ckpt = torch.load(perception_checkpoint, map_location="cpu", weights_only=False)
    classifier.load_state_dict(cls_ckpt["classifier_state_dict"])

    # Freeze fusion + classifier (frozen in cognition per skeleton spec)
    for p in list(fusion.parameters()) + list(classifier.parameters()):
        p.requires_grad = False
    fusion.eval()
    classifier.eval()

    # Load LLM with LoRA
    model_name = llm_cfg.base_model.name
    logger.info("Loading LLM: %s", model_name)
    quant_cfg = _make_bnb_config(llm_cfg.base_model.quantization)
    lm_kwargs: dict = {"device_map": "auto"}
    if quant_cfg is not None:
        lm_kwargs["quantization_config"] = quant_cfg
    else:
        lm_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name, **lm_kwargs)

    lora_cfg_ns = ccfg_train.lora
    if getattr(lora_cfg_ns, "enabled", True):
        lora_config = LoraConfig(
            r=lora_cfg_ns.rank,
            lora_alpha=lora_cfg_ns.alpha,
            target_modules=list(lora_cfg_ns.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()

    from vie_gameemo.llm.modal_adapter import ModalAdapter
    llm_hidden_size = llm.config.hidden_size
    llm_adapter = ModalAdapter(d_fusion=fcfg.d_model, d_llm=llm_hidden_size).to(device)

    # If Stage 2a checkpoint exists, warm-start adapter + LoRA from it
    if llm_perception_checkpoint and Path(llm_perception_checkpoint).exists():
        lp_ckpt = torch.load(llm_perception_checkpoint, map_location="cpu", weights_only=False)
        if "llm_adapter" in lp_ckpt:
            llm_adapter.load_state_dict(lp_ckpt["llm_adapter"], strict=False)
            logger.info("Loaded ModalAdapter from Stage 2a: %s", llm_perception_checkpoint)
        if "llm_peft" in lp_ckpt and lp_ckpt["llm_peft"] is not None:
            llm.load_state_dict(lp_ckpt["llm_peft"], strict=False)
            logger.info("Loaded LLM LoRA from Stage 2a: %s", llm_perception_checkpoint)
    else:
        logger.info("No Stage 2a checkpoint — starting ModalAdapter + LoRA from scratch")

    cls_criterion = FocalLoss(gamma=ccfg.loss.focal.gamma)

    # Balanced sampler for cognition
    train_loader = _maybe_balanced_sampler(train_loader, cfg)

    trainable_params = list(llm.parameters()) + list(llm_adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=ccfg_train.learning_rate.llm,
        weight_decay=getattr(pcfg, "weight_decay", 0.01),
    )

    n_steps = len(train_loader) * ccfg_train.epochs
    warmup_steps = max(1, int(n_steps * getattr(pcfg, "warmup_ratio", 0.1)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "cognition_best.pt"

    alpha = ccfg_train.loss_weights.classification
    beta = ccfg_train.loss_weights.reasoning_lm

    logger.info("Cognition training: %d epochs, alpha=%.2f, beta=%.2f", ccfg_train.epochs, alpha, beta)
    best_metric = float("-inf")

    for epoch in range(ccfg_train.epochs):
        fusion.eval()
        classifier.eval()
        llm.train()
        llm_adapter.train()

        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            context = batch["context"].to(device)
            text = batch["text"].to(device)
            gt_labels = batch["label"].to(device)
            has_face = batch.get("has_face")
            if has_face is not None:
                has_face = has_face.to(device)

            with torch.no_grad():
                fused = fusion(audio, face, context, text, has_face=has_face)
                if isinstance(fused, tuple):
                    fused = fused[0]
                cls_logits = classifier(fused)

            cls_loss = cls_criterion(cls_logits, gt_labels)

            # Multi-task LM loss: random task per sample (like v2)
            B = fused.shape[0]
            import random as _rnd

            task_prompts = []
            task_targets = []
            task_audio = audio.clone()
            task_face = face.clone()
            task_context = context.clone()
            task_text = text.clone()

            for i in range(B):
                gt_name = _LABEL_NAMES[gt_labels[i].item()]
                task = _rnd.choice(_COGNITION_TASKS)

                if task == "audio":
                    audio_desc = batch.get("audio_desc", [""] * B)
                    desc = audio_desc[i] if isinstance(audio_desc, (list, tuple)) else ""
                    if not desc:
                        task = "emotion"
                    else:
                        task_prompts.append(_TASK_AUDIO_PROMPT)
                        task_targets.append(desc)
                        task_face[i] = 0
                        task_context[i] = 0
                        task_text[i] = 0
                        continue

                if task == "visual":
                    visual_desc = batch.get("visual_desc", [""] * B)
                    desc = visual_desc[i] if isinstance(visual_desc, (list, tuple)) else ""
                    if not desc:
                        task = "emotion"
                    else:
                        task_prompts.append(_TASK_VISUAL_PROMPT)
                        task_targets.append(desc)
                        task_audio[i] = 0
                        task_text[i] = 0
                        continue

                if task == "reasoning":
                    reasoning = batch.get("reasoning_text", [""] * B)
                    r = reasoning[i] if isinstance(reasoning, (list, tuple)) else ""
                    if not r:
                        task = "emotion"
                    else:
                        task_prompts.append(_TASK_REASONING_PROMPT.format(label=gt_name))
                        task_targets.append(r)
                        continue

                # Default: emotion label prediction
                labels_str = ", ".join(_LABEL_NAMES)
                task_prompts.append(_LLM_PERCEPTION_PROMPT.format(labels=labels_str))
                task_targets.append(f"<answer>{gt_name}</answer>")

            # Re-fuse with task-specific zeroed modalities
            with torch.no_grad():
                task_fused = fusion(task_audio, task_face, task_context, task_text, has_face=has_face)
                if isinstance(task_fused, tuple):
                    task_fused = task_fused[0]

            soft_tokens, soft_mask = llm_adapter(
                task_fused, audio=task_audio, face=task_face, context=task_context,
                text=task_text, has_face=has_face,
            )
            n_soft = soft_tokens.shape[1]

            full_texts = [p + "\n" + t for p, t in zip(task_prompts, task_targets)]
            prompt_only = tokenizer(
                task_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(device)
            full = tokenizer(
                full_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)

            embed_fn = llm.get_input_embeddings()
            full_embeds = embed_fn(full["input_ids"])

            inputs_embeds = torch.cat([soft_tokens, full_embeds], dim=1)
            attn_mask = torch.cat([soft_mask, full["attention_mask"]], dim=1)

            lm_labels = full["input_ids"].clone()
            for i in range(B):
                prompt_len = prompt_only["attention_mask"][i].sum().item()
                lm_labels[i, :prompt_len] = -100
            lm_labels = torch.cat([
                torch.full((B, n_soft), -100, device=device, dtype=torch.long),
                lm_labels,
            ], dim=1)

            lm_out = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=lm_labels,
            )
            lm_loss = lm_out.loss

            total_loss = alpha * cls_loss + beta * lm_loss
            total_loss.backward()
            epoch_loss += total_loss.item()

            grad_accum = getattr(ccfg_train, "gradient_accumulation", 1)
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        avg_loss = epoch_loss / max(1, len(train_loader))

        # Evaluate LLM's own predictions (label accuracy + format compliance)
        llm_metrics = _eval_llm_metrics(
            fusion, llm_adapter, llm, tokenizer, val_loader,
            device, use_mlp_hint=False, classifier=classifier, n_samples=50,
        )
        llm_f1 = llm_metrics["macro_f1"]
        logger.info(
            "Epoch %d/%d | loss=%.4f | llm_macro_f1=%.4f | llm_uar=%.4f | format=%.2f",
            epoch + 1, ccfg_train.epochs, avg_loss,
            llm_f1, llm_metrics["uar"], llm_metrics["format_rate"],
        )
        if llm_metrics["per_class_f1"]:
            for cls_name, f1_val in llm_metrics["per_class_f1"].items():
                logger.info("  %s: F1=%.4f", cls_name, f1_val)

        if llm_f1 > best_metric:
            best_metric = llm_f1
            torch.save({
                "llm_adapter": llm_adapter.state_dict(),
                "llm_peft": llm.state_dict() if hasattr(llm, "peft_config") else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "llm_metrics": llm_metrics,
            }, best_ckpt)
            logger.info("New best cognition model saved (llm_macro_f1=%.4f)", llm_f1)

    logger.info("Cognition training done. Best llm_macro_f1=%.4f", best_metric)
    return best_ckpt


def build_llm_input_embeds(
    u_fusion: torch.Tensor,
    llm_adapter: torch.nn.Module,
    llm_embed_layer: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Build inputs_embeds for the LLM: prompt tokens + projected fusion features.

    Args:
        u_fusion: (B, T, 768) from frozen fusion module.
        llm_adapter: Linear projector 768 → llm_hidden_size.
        llm_embed_layer: LLM's input embedding layer.
        prompt_token_ids: (B, P) prompt token IDs.

    Returns:
        (B, P + T, llm_hidden_size) embedding sequence.
    """
    prompt_embeds = llm_embed_layer(prompt_token_ids)           # (B, P, H)
    vision_embeds = llm_adapter(u_fusion)                       # (B, T, H)
    return torch.cat([prompt_embeds, vision_embeds], dim=1)     # (B, P+T, H)


def _make_bnb_config(quantization: str):
    """Build BitsAndBytesConfig for quantization."""
    try:
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        pass
    return None
