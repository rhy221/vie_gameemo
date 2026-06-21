"""Stage 2 — Cognition trainer (joint recognition + reasoning).

Loads the Stage 1 checkpoint (perception), FREEZES fusion + classifier,
and trains an LLM (Qwen2.5-7B with LoRA) to generate reasoning explanations
alongside classification.

Loss = α * L_classification + β * L_reasoning_LM
    where L_reasoning_LM is the standard language modeling loss on the
    multi-agent-generated reasoning text (cached during Stage 0).
"""

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_cognition(
    cfg: SimpleNamespace,
    perception_checkpoint: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    resume_from: Path | None = None,
) -> Path:
    """Train Stage 2 — Cognition (joint cls + reasoning).

    Args:
        cfg: Full config namespace.
        perception_checkpoint: Stage 1 checkpoint (fusion + classifier).
        train_loader: DataLoader with batches including reasoning targets.
        val_loader: Validation DataLoader.
        device: Torch device.
        resume_from: Optional cognition checkpoint to resume from.

    Returns:
        Path to best checkpoint.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.fusion import get_fusion
    from vie_gameemo.training.losses import FocalLoss
    from vie_gameemo.training.perception import TrainingState, evaluate, _save_checkpoint

    ccfg_train = cfg.training.cognition
    pcfg = cfg.training.perception
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # Build and load frozen fusion + classifier from Stage 1
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

    ckpt = torch.load(perception_checkpoint, map_location="cpu")
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    # Freeze fusion + classifier
    for p in list(fusion.parameters()) + list(classifier.parameters()):
        p.requires_grad = False
    logger.info("Loaded and froze perception checkpoint from %s", perception_checkpoint)

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

    # Modal adapter: project fusion dim → LLM embedding space (Emotion-LLaMAv2 pattern)
    from vie_gameemo.llm.modal_adapter import ModalAdapter
    llm_hidden_size = llm.config.hidden_size
    llm_adapter = ModalAdapter(d_fusion=fcfg.d_model, d_llm=llm_hidden_size).to(device)

    cls_criterion = FocalLoss(gamma=ccfg.loss.focal.gamma)

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
            labels = batch["label"].to(device)
            has_face = batch.get("has_face")
            if has_face is not None:
                has_face = has_face.to(device)

            with torch.no_grad():
                fused = fusion(audio, face, context, text, has_face=has_face)
                if isinstance(fused, tuple):
                    fused = fused[0]
                cls_logits = classifier(fused)

            cls_loss = cls_criterion(cls_logits, labels)

            # Language modeling loss: soft token (from modal adapter) + reasoning text
            lm_loss = torch.tensor(0.0, device=device)
            reasoning_texts = batch.get("reasoning_text")
            if reasoning_texts:
                B = fused.shape[0]
                # Pool fused sequence → 1 soft token per sample, project to LLM dim
                soft_token = llm_adapter(fused).mean(dim=1, keepdim=True)  # (B, 1, H)

                inputs = tokenizer(
                    list(reasoning_texts),
                    return_tensors="pt",
                    truncation=True,
                    max_length=255,
                    padding=True,
                ).to(device)

                embed_fn = llm.get_input_embeddings()
                text_embeds = embed_fn(inputs["input_ids"])          # (B, L, H)

                # Inject soft token before text: [soft_token | text_tokens]
                inputs_embeds = torch.cat([soft_token, text_embeds], dim=1)  # (B, 1+L, H)
                attn_mask = torch.cat([
                    torch.ones(B, 1, device=device, dtype=torch.long),
                    inputs["attention_mask"],
                ], dim=1)
                # Mask soft token from language modeling loss (-100 = ignore)
                labels = torch.cat([
                    torch.full((B, 1), -100, device=device, dtype=torch.long),
                    inputs["input_ids"],
                ], dim=1)

                lm_out = llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
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
        val_metrics = evaluate(fusion=fusion, classifier=classifier,
                               loader=val_loader, device=device, n_classes=ccfg.n_classes)
        macro_f1 = val_metrics["macro_f1"]
        logger.info("Epoch %d/%d | loss=%.4f | val_macro_f1=%.4f",
                    epoch + 1, ccfg_train.epochs, avg_loss, macro_f1)

        if macro_f1 > best_metric:
            best_metric = macro_f1
            torch.save({
                "llm_adapter": llm_adapter.state_dict(),
                "llm_peft": llm.state_dict() if hasattr(llm, "peft_config") else None,
                "epoch": epoch,
                "best_metric": best_metric,
            }, best_ckpt)
            logger.info("New best cognition model saved (macro_f1=%.4f)", macro_f1)

    logger.info("Cognition training done. Best macro_f1=%.4f", best_metric)
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
