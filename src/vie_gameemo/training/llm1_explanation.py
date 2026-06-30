"""LLM-1 Faithful Explainer training (2-stage).

Stage A — Alignment:
    Freeze LLM + MLP + encoders + fusion.
    Train ModalAdapter (incl. proj_penult) + g_head.
    Loss = L_LM + λ_rec * L_rec [+ λ_kl * L_kl]

Stage B — LoRA fine-tune (optional):
    Add LoRA to LLM, keep MLP + encoders + fusion frozen.
    Continue training ModalAdapter + g_head + LLM LoRA.
    Same loss, lower LR for LoRA.

Target text format:
    "Cues: face: ...; voice: ...; scene: ...; text: .... Emotion: {MLP_label}."
"""

import logging
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_LABEL_NAMES = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


class GHead(nn.Module):
    """Small MLP: mean-pooled raw modality tokens → attribute vector.

    Guards against the LLM shortcut of ignoring tap A (raw tokens).
    """

    def __init__(self, d_input: int = 768, hidden_dim: int = 128, n_attrs: int = 15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_attrs),
        )

    def forward(self, z_raw: torch.Tensor) -> torch.Tensor:
        return self.net(z_raw)


def collate_fn_llm1(batch: list[dict]) -> dict:
    """Extended collate that preserves metadata for LLM-1 training."""
    from vie_gameemo.data.dataset import _pad_and_stack

    clip_ids = [b["clip_id"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    has_face = torch.tensor([b["has_face"] for b in batch], dtype=torch.bool)

    audio = _pad_and_stack([b["audio"] for b in batch])
    face = _pad_and_stack([b["face"] for b in batch])
    context = _pad_and_stack([b["context"] for b in batch])
    text = _pad_and_stack([b["text"] for b in batch])

    return {
        "audio": audio,
        "face": face,
        "context": context,
        "text": text,
        "label": labels,
        "has_face": has_face,
        "clip_id": clip_ids,
        "transcript": [b.get("transcript", "") for b in batch],
        "source_language": [b.get("source_language", "vi") for b in batch],
    }


def _mean_pool_raw(audio, face, context, text):
    """Mean-pool raw modality tokens → (B, 768) for g_head."""
    parts = []
    for t in [audio, face, context, text]:
        parts.append(t.mean(dim=1))
    return torch.stack(parts, dim=1).mean(dim=1)


def _modality_dropout(audio, face, context, text, p: float = 0.3):
    """Randomly zero one modality per sample (augmentation)."""
    B = audio.shape[0]
    for i in range(B):
        if random.random() < p:
            drop = random.randint(0, 3)
            if drop == 0:
                audio[i] = 0
            elif drop == 1:
                face[i] = 0
            elif drop == 2:
                context[i] = 0
            else:
                text[i] = 0
    return audio, face, context, text


def train_llm1_stage_a(
    cfg: SimpleNamespace,
    perception_checkpoint: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Path:
    """Stage A: alignment — train ModalAdapter + g_head, freeze everything else."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.fusion import get_fusion
    from vie_gameemo.llm.cue_extractor import CueExtractor
    from vie_gameemo.llm.modal_adapter import ModalAdapter

    tcfg = cfg.training.llm1_explanation
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # --- Load frozen components ---
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

    ckpt = torch.load(perception_checkpoint, map_location="cpu")
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    for p in list(fusion.parameters()) + list(classifier.parameters()):
        p.requires_grad = False
    fusion.eval()
    classifier.eval()

    # --- Load LLM (frozen in stage A) ---
    model_name = getattr(llm_cfg.base_model, "fallback", llm_cfg.base_model.name)
    logger.info("Loading LLM (frozen): %s", model_name)
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

    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    # --- Trainable components ---
    llm_hidden = llm.config.hidden_size
    adapter = ModalAdapter(
        d_fusion=fcfg.d_model, d_llm=llm_hidden,
        d_penult=ccfg.hidden_dim,
    ).to(device)

    g_head_cfg = tcfg.g_head
    g_head = GHead(
        d_input=fcfg.d_model,
        hidden_dim=g_head_cfg.hidden_dim,
        n_attrs=g_head_cfg.n_attrs,
    ).to(device)

    cue_extractor = CueExtractor(cache_dir=cfg.paths.cache)

    trainable_params = list(adapter.parameters()) + list(g_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter.parameters(), "lr": tcfg.learning_rate.adapter},
            {"params": g_head.parameters(), "lr": tcfg.learning_rate.g_head},
        ],
        weight_decay=0.01,
    )

    n_epochs = tcfg.epochs_a
    n_steps = len(train_loader) * n_epochs
    warmup_steps = max(1, int(n_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / warmup_steps, 0.5 * (1 + math.cos(math.pi * max(0, s - warmup_steps) / max(1, n_steps - warmup_steps)))),
    )

    lambda_kl = tcfg.loss.lambda_kl
    lambda_rec = tcfg.loss.lambda_rec
    kl_temp = tcfg.loss.kl_temperature
    grad_accum = getattr(tcfg, "gradient_accumulation", 4)

    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "llm1_explanation_best.pt"

    logger.info("LLM-1 Stage A: %d epochs, λ_kl=%.2f, λ_rec=%.2f", n_epochs, lambda_kl, lambda_rec)
    best_metric = float("-inf")

    for epoch in range(n_epochs):
        adapter.train()
        g_head.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            loss = _compute_loss(
                batch, fusion, classifier, adapter, g_head, llm, tokenizer,
                cue_extractor, device, lambda_kl, lambda_rec, kl_temp,
                modality_dropout_p=0.3,
            )

            loss = loss / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        avg_loss = epoch_loss / max(1, len(train_loader))
        metrics = _eval_agreement(
            fusion, classifier, adapter, llm, tokenizer,
            cue_extractor, val_loader, device, n_samples=50,
        )

        logger.info(
            "Epoch %d/%d | loss=%.4f | agreement=%.4f | format=%.2f",
            epoch + 1, n_epochs, avg_loss,
            metrics["agreement"], metrics["format_rate"],
        )

        if metrics["agreement"] > best_metric:
            best_metric = metrics["agreement"]
            torch.save({
                "llm_adapter": adapter.state_dict(),
                "g_head": g_head.state_dict(),
                "llm_peft": None,
                "epoch": epoch,
                "stage": "a",
                "best_metric": best_metric,
                "metrics": metrics,
            }, best_ckpt)
            logger.info("New best Stage A (agreement=%.4f)", best_metric)

    logger.info("Stage A done. Best agreement=%.4f", best_metric)
    return best_ckpt


def train_llm1_stage_b(
    cfg: SimpleNamespace,
    perception_checkpoint: Path,
    stage_a_checkpoint: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Path:
    """Stage B: LoRA fine-tune — add LoRA to LLM, continue training."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.classifiers.mlp import EmotionClassifier
    from vie_gameemo.fusion import get_fusion
    from vie_gameemo.llm.cue_extractor import CueExtractor
    from vie_gameemo.llm.modal_adapter import ModalAdapter

    tcfg = cfg.training.llm1_explanation
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # --- Frozen perception ---
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

    ckpt = torch.load(perception_checkpoint, map_location="cpu")
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    for p in list(fusion.parameters()) + list(classifier.parameters()):
        p.requires_grad = False
    fusion.eval()
    classifier.eval()

    # --- LLM with LoRA ---
    model_name = getattr(llm_cfg.base_model, "fallback", llm_cfg.base_model.name)
    logger.info("Loading LLM with LoRA: %s", model_name)
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

    lora_cfg = tcfg.lora
    lora_config = LoraConfig(
        r=lora_cfg.rank, lora_alpha=lora_cfg.alpha,
        target_modules=list(lora_cfg.target_modules),
        bias="none", task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # --- Load Stage A adapter + g_head ---
    llm_hidden = llm.config.hidden_size
    adapter = ModalAdapter(
        d_fusion=fcfg.d_model, d_llm=llm_hidden,
        d_penult=ccfg.hidden_dim,
    ).to(device)

    g_head_cfg = tcfg.g_head
    g_head = GHead(
        d_input=fcfg.d_model,
        hidden_dim=g_head_cfg.hidden_dim,
        n_attrs=g_head_cfg.n_attrs,
    ).to(device)

    sa_ckpt = torch.load(stage_a_checkpoint, map_location="cpu")
    adapter.load_state_dict(sa_ckpt["llm_adapter"], strict=False)
    g_head.load_state_dict(sa_ckpt["g_head"])
    logger.info("Loaded Stage A checkpoint: %s", stage_a_checkpoint)

    cue_extractor = CueExtractor(cache_dir=cfg.paths.cache)

    trainable_params = list(adapter.parameters()) + list(g_head.parameters()) + list(llm.parameters())
    optimizer = torch.optim.AdamW([
        {"params": adapter.parameters(), "lr": tcfg.learning_rate.adapter},
        {"params": g_head.parameters(), "lr": tcfg.learning_rate.g_head},
        {"params": llm.parameters(), "lr": tcfg.learning_rate.lora},
    ], weight_decay=0.01)

    n_epochs = tcfg.epochs_b
    n_steps = len(train_loader) * n_epochs
    warmup_steps = max(1, int(n_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / warmup_steps, 0.5 * (1 + math.cos(math.pi * max(0, s - warmup_steps) / max(1, n_steps - warmup_steps)))),
    )

    lambda_kl = tcfg.loss.lambda_kl
    lambda_rec = tcfg.loss.lambda_rec
    kl_temp = tcfg.loss.kl_temperature
    grad_accum = getattr(tcfg, "gradient_accumulation", 4)

    ckpt_dir = Path(cfg.paths.checkpoints)
    best_ckpt = ckpt_dir / "llm1_explanation_best.pt"
    best_metric = sa_ckpt.get("best_metric", 0.0)
    patience = getattr(tcfg.early_stopping, "patience", 5)
    no_improve = 0

    logger.info("LLM-1 Stage B: %d epochs, LoRA rank=%d", n_epochs, lora_cfg.rank)

    for epoch in range(n_epochs):
        adapter.train()
        g_head.train()
        llm.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            loss = _compute_loss(
                batch, fusion, classifier, adapter, g_head, llm, tokenizer,
                cue_extractor, device, lambda_kl, lambda_rec, kl_temp,
                modality_dropout_p=0.3,
            )
            loss = loss / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        avg_loss = epoch_loss / max(1, len(train_loader))
        metrics = _eval_agreement(
            fusion, classifier, adapter, llm, tokenizer,
            cue_extractor, val_loader, device, n_samples=50,
        )

        logger.info(
            "Epoch %d/%d | loss=%.4f | agreement=%.4f | format=%.2f",
            epoch + 1, n_epochs, avg_loss,
            metrics["agreement"], metrics["format_rate"],
        )

        if metrics["agreement"] > best_metric:
            best_metric = metrics["agreement"]
            no_improve = 0
            torch.save({
                "llm_adapter": adapter.state_dict(),
                "g_head": g_head.state_dict(),
                "llm_peft": llm.state_dict(),
                "epoch": epoch,
                "stage": "b",
                "best_metric": best_metric,
                "metrics": metrics,
            }, best_ckpt)
            logger.info("New best Stage B (agreement=%.4f)", best_metric)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Stage B done. Best agreement=%.4f", best_metric)
    return best_ckpt


# ------------------------------------------------------------------
# Loss computation
# ------------------------------------------------------------------

def _compute_loss(
    batch, fusion, classifier, adapter, g_head, llm, tokenizer,
    cue_extractor, device, lambda_kl, lambda_rec, kl_temp,
    modality_dropout_p=0.0,
):
    """Compute L = L_LM + λ_kl * L_kl + λ_rec * L_rec."""
    audio = batch["audio"].to(device)
    face = batch["face"].to(device)
    context = batch["context"].to(device)
    text_feat = batch["text"].to(device)
    has_face = batch["has_face"].to(device)
    clip_ids = batch["clip_id"]
    transcripts = batch["transcript"]
    B = audio.shape[0]

    # Modality dropout augmentation
    if modality_dropout_p > 0:
        audio, face, context, text_feat = _modality_dropout(
            audio.clone(), face.clone(), context.clone(), text_feat.clone(),
            p=modality_dropout_p,
        )

    # --- Frozen forward ---
    with torch.no_grad():
        fused = fusion(audio, face, context, text_feat, has_face=has_face)
        if isinstance(fused, tuple):
            fused = fused[0]
        logits, penult = classifier(fused, return_penultimate=True)

    # --- Build targets ---
    prompts = []
    targets = []
    attr_vecs = []

    for i in range(B):
        mlp_idx = int(logits[i].argmax().item())
        mlp_label = _LABEL_NAMES[mlp_idx]

        cue_text, attr_vec = cue_extractor.extract(
            clip_ids[i], transcripts[i], bool(has_face[i].item()),
        )
        target = f"Cues: {cue_text}. Emotion: {mlp_label}."
        prompt = "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được và xác định cảm xúc."

        prompts.append(prompt)
        targets.append(target)
        attr_vecs.append(attr_vec)

    attr_tensor = torch.stack(attr_vecs).to(device)

    # --- Soft tokens ---
    soft_tokens, soft_mask = adapter(
        fused, penult=penult, audio=audio, face=face,
        context=context, text=text_feat, has_face=has_face,
    )
    n_soft = soft_tokens.shape[1]

    # --- L_LM: causal LM loss on target text ---
    full_texts = [p + "\n" + t for p, t in zip(prompts, targets)]
    prompt_enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=128,
    ).to(device)
    full_enc = tokenizer(
        full_texts, return_tensors="pt", padding=True, truncation=True, max_length=256,
    ).to(device)

    embed_fn = llm.get_input_embeddings()
    full_embeds = embed_fn(full_enc["input_ids"])
    inputs_embeds = torch.cat([soft_tokens, full_embeds], dim=1)
    attn_mask = torch.cat([soft_mask, full_enc["attention_mask"]], dim=1)

    lm_labels = full_enc["input_ids"].clone()
    for i in range(B):
        prompt_len = prompt_enc["attention_mask"][i].sum().item()
        lm_labels[i, :prompt_len] = -100
    lm_labels = torch.cat([
        torch.full((B, n_soft), -100, device=device, dtype=torch.long),
        lm_labels,
    ], dim=1)

    lm_out = llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=lm_labels)
    loss_lm = lm_out.loss

    total_loss = loss_lm

    # --- L_rec: g_head reconstruction loss ---
    if lambda_rec > 0:
        z_raw = _mean_pool_raw(audio, face, context, text_feat)
        attr_pred = g_head(z_raw)
        loss_rec = F.smooth_l1_loss(attr_pred, attr_tensor)
        total_loss = total_loss + lambda_rec * loss_rec

    # --- L_kl: soft distillation from MLP confidence ---
    if lambda_kl > 0:
        mlp_soft = F.softmax(logits / kl_temp, dim=-1).detach()
        lm_logits_at_emotion = _extract_emotion_logprobs(lm_out.logits, lm_labels, tokenizer)
        if lm_logits_at_emotion is not None:
            lm_soft = F.log_softmax(lm_logits_at_emotion / kl_temp, dim=-1)
            loss_kl = F.kl_div(lm_soft, mlp_soft, reduction="batchmean") * (kl_temp ** 2)
            total_loss = total_loss + lambda_kl * loss_kl

    return total_loss


def _extract_emotion_logprobs(lm_logits, labels, tokenizer):
    """Try to extract logits at positions corresponding to emotion label tokens.

    Returns None if extraction fails (skip L_kl in that case).
    """
    try:
        label_token_ids = []
        for name in _LABEL_NAMES:
            ids = tokenizer.encode(name, add_special_tokens=False)
            if ids:
                label_token_ids.append(ids[0])
        if len(label_token_ids) != len(_LABEL_NAMES):
            return None

        B = lm_logits.shape[0]
        emotion_logits = []
        for i in range(B):
            valid = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                return None
            last_pos = valid[-1].item()
            token_logits = lm_logits[i, last_pos, label_token_ids]
            emotion_logits.append(token_logits)
        return torch.stack(emotion_logits)
    except Exception:
        return None


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def _eval_agreement(
    fusion, classifier, adapter, llm, tokenizer,
    cue_extractor, val_loader, device, n_samples=50,
):
    """Evaluate LLM-1 agreement with MLP and format compliance."""
    import re

    adapter.eval()
    total = 0
    n_agree = 0
    n_format = 0

    prompt = "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được và xác định cảm xúc."

    for batch in val_loader:
        if total >= n_samples:
            break

        audio = batch["audio"].to(device)
        face = batch["face"].to(device)
        context = batch["context"].to(device)
        text_feat = batch["text"].to(device)
        has_face = batch["has_face"].to(device)
        B = audio.shape[0]

        with torch.no_grad():
            fused = fusion(audio, face, context, text_feat, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]
            logits, penult = classifier(fused, return_penultimate=True)

            soft_tokens, soft_mask = adapter(
                fused, penult=penult, audio=audio, face=face,
                context=context, text=text_feat, has_face=has_face,
            )

            for i in range(min(B, n_samples - total)):
                mlp_idx = int(logits[i].argmax().item())
                mlp_label = _LABEL_NAMES[mlp_idx]

                text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                text_embeds = llm.get_input_embeddings()(text_ids)
                inputs_embeds = torch.cat([soft_tokens[i:i+1], text_embeds], dim=1)

                out_ids = llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                raw = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

                emotion_match = re.search(r"Emotion:\s*(\w+)", raw)
                if emotion_match:
                    n_format += 1
                    predicted = emotion_match.group(1).strip().lower()
                    if predicted == mlp_label:
                        n_agree += 1

                cue_match = re.search(r"Cues:", raw)
                if cue_match and emotion_match:
                    n_format += 0  # already counted

                total += 1

    adapter.train()

    return {
        "agreement": n_agree / max(1, total),
        "format_rate": n_format / max(1, total),
        "n_samples": total,
    }


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _make_bnb_config(quantization: str):
    try:
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        pass
    return None
