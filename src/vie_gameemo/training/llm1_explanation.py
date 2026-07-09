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
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vie_gameemo.training.losses import modality_dropout as _modality_dropout

# PyTorch < 2.1 lacks nn.Module.set_submodule, which newer transformers/bitsandbytes
# call during 4-bit quantization. Patch it if missing so quantization still works.
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.split(".")
        parent = self
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)
    nn.Module.set_submodule = _set_submodule  # type: ignore[method-assign]

logger = logging.getLogger(__name__)

_LABEL_NAMES = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


class GHeadPerModality(nn.Module):
    """Per-modality reconstruction heads — each reads ONLY its own raw token (tap A).

    Replaces the old global-pool GHead. Separate heads prevent strong modalities
    from masking weaker ones and ensure each raw token retains its own cue signal.

    Args:
        d_input: Raw modality embedding dim (768).
        hidden_dim: Hidden dim for each head.
        n_face: Attribute dims for face (EAR/MAR/brow/yaw/pitch = 5).
        n_voice: Attribute dims for voice (f0/rms/rate = 3).
        n_motion: Attribute dims for motion cue (energy/impact/period = 3).
            Only used when has_context=True (pose branch).
        n_text: Attribute dims for text (exclaim/neg/game/words = 4).
        has_context: Whether a context (motion) head is included.
            Set False for vit_imagenet branch (no motion cue).
    """

    def __init__(
        self,
        d_input: int = 768,
        hidden_dim: int = 128,
        n_face: int = 5,
        n_voice: int = 3,
        n_motion: int = 3,
        n_text: int = 4,
        has_context: bool = True,
        text_dim: int | None = None,
        audio_dim: int | None = None,
        face_dim: int | None = None,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.has_context = has_context

        def _head(in_dim: int, n_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_out),
            )

        self.face_head = _head(face_dim or d_input, n_face)
        self.voice_head = _head(audio_dim or d_input, n_voice)
        self.text_head = _head(text_dim or d_input, n_text)
        self.motion_head = _head(context_dim or d_input, n_motion) if has_context else None

    def forward(
        self,
        face: torch.Tensor,
        audio: torch.Tensor,
        context: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Reconstruct per-modality attribute sub-vectors.

        Args:
            face: (B, T, 768) raw face encoder tokens (tap A only — NOT penult/fusion).
            audio: (B, T, 768) raw audio encoder tokens.
            context: (B, T, 768) raw context encoder tokens.
            text: (B, T, 768) raw text encoder tokens.

        Returns:
            (face_pred, voice_pred, motion_pred_or_None, text_pred)
        """
        face_pred = self.face_head(face.mean(dim=1))
        voice_pred = self.voice_head(audio.mean(dim=1))
        text_pred = self.text_head(text.mean(dim=1))
        motion_pred = self.motion_head(context.mean(dim=1)) if self.has_context else None
        return face_pred, voice_pred, motion_pred, text_pred


def _infer_fusion_dims(ckpt: dict) -> dict:
    """Read per-modality input dims from saved fusion weights.

    Inspects mlp_<modal>.weight shape in the checkpoint's fusion_state_dict.
    Shape is (d_model, in_dim), so in_dim = weight.shape[1].
    Used to reconstruct the fusion module with the exact architecture that
    was used during perception training (avoids config/checkpoint mismatch).
    """
    dims: dict[str, int] = {}
    sd = ckpt.get("fusion_state_dict", {})
    for modal in ("text", "audio", "face", "context"):
        w = sd.get(f"mlp_{modal}.weight")
        if w is not None:
            dims[f"{modal}_dim"] = w.shape[1]
    return dims


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



def train_llm1_stage_a(
    cfg: SimpleNamespace,
    perception_checkpoint: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Path:
    """Stage A: alignment — train ModalAdapter + g_head, freeze everything else."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.classifiers import get_classifier
    from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
    from vie_gameemo.llm.cue_extractor import CueExtractor
    from vie_gameemo.llm.modal_adapter import ModalAdapter

    tcfg = cfg.training.llm1_explanation
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # --- Load frozen components ---
    # Load checkpoint first so we can infer the exact per-modality dims that
    # were used during perception training (e.g. text_dim=1024 for CafeBERT).
    # This avoids a size-mismatch when the config value differs from the saved weights.
    ckpt = torch.load(perception_checkpoint, map_location="cpu", weights_only=True)
    _dim_kwargs = {**modality_dim_kwargs(fcfg), **_infer_fusion_dims(ckpt)}
    fusion = get_fusion(
        fcfg.type, d_model=fcfg.d_model, n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
        skip_mlp_if_matched=getattr(fcfg, "skip_mlp_if_matched", False),
        **_dim_kwargs,
    ).to(device)
    classifier = get_classifier(
        ccfg, d_model=fcfg.d_model, device=device,
        classifier_type=ckpt.get("classifier_type"),
    )

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
    # text_dim excluded: text goes into prompt as raw text, not soft token
    _adapter_dims = {k: v for k, v in modality_dim_kwargs(fcfg).items() if k != "text_dim"}
    adapter = ModalAdapter(
        d_fusion=fcfg.d_model, d_llm=llm_hidden,
        d_penult=ccfg.hidden_dim,
        **_adapter_dims,
    ).to(device)

    g_head_cfg = tcfg.g_head
    ctx_type = getattr(getattr(cfg, "visual_encoder", SimpleNamespace()), "context_encoder", SimpleNamespace())
    ctx_encoder_type = getattr(ctx_type, "type", "vit_imagenet")
    has_context = ctx_encoder_type == "pose"
    g_head = GHeadPerModality(
        d_input=fcfg.d_model,
        hidden_dim=g_head_cfg.hidden_dim,
        has_context=has_context,
        **modality_dim_kwargs(fcfg),
    ).to(device)

    cue_extractor = CueExtractor(cache_dir=cfg.paths.cache, context_encoder_type=ctx_encoder_type)

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

    n_steps_per_epoch = len(train_loader)
    log_every = max(1, n_steps_per_epoch // 10)

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

            if (step + 1) % log_every == 0 or (step + 1) == n_steps_per_epoch:
                avg_so_far = epoch_loss / (step + 1)
                logger.info(
                    "  [A] Epoch %d/%d | step %d/%d | loss=%.4f",
                    epoch + 1, n_epochs, step + 1, n_steps_per_epoch, avg_so_far,
                )

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

    from vie_gameemo.classifiers import get_classifier
    from vie_gameemo.fusion import get_fusion, modality_dim_kwargs
    from vie_gameemo.llm.cue_extractor import CueExtractor
    from vie_gameemo.llm.modal_adapter import ModalAdapter

    tcfg = cfg.training.llm1_explanation
    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # --- Frozen perception ---
    ckpt = torch.load(perception_checkpoint, map_location="cpu", weights_only=True)
    _dim_kwargs = {**modality_dim_kwargs(fcfg), **_infer_fusion_dims(ckpt)}
    fusion = get_fusion(
        fcfg.type, d_model=fcfg.d_model, n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
        skip_mlp_if_matched=getattr(fcfg, "skip_mlp_if_matched", False),
        **_dim_kwargs,
    ).to(device)
    classifier = get_classifier(
        ccfg, d_model=fcfg.d_model, device=device,
        classifier_type=ckpt.get("classifier_type"),
    )

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

    # Recompute activations during backward instead of storing all 28-layer activations.
    # Saves ~50% activation VRAM at ~30% extra compute cost — essential for 7B + LoRA on 32GB.
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- Load Stage A adapter + g_head ---
    llm_hidden = llm.config.hidden_size
    _adapter_dims = {k: v for k, v in modality_dim_kwargs(fcfg).items() if k != "text_dim"}
    adapter = ModalAdapter(
        d_fusion=fcfg.d_model, d_llm=llm_hidden,
        d_penult=ccfg.hidden_dim,
        **_adapter_dims,
    ).to(device)

    g_head_cfg = tcfg.g_head
    ctx_type = getattr(getattr(cfg, "visual_encoder", SimpleNamespace()), "context_encoder", SimpleNamespace())
    ctx_encoder_type = getattr(ctx_type, "type", "vit_imagenet")
    has_context = ctx_encoder_type == "pose"
    g_head = GHeadPerModality(
        d_input=fcfg.d_model,
        hidden_dim=g_head_cfg.hidden_dim,
        has_context=has_context,
        **modality_dim_kwargs(fcfg),
    ).to(device)

    sa_ckpt = torch.load(stage_a_checkpoint, map_location="cpu", weights_only=True)
    adapter.load_state_dict(sa_ckpt["llm_adapter"], strict=False)
    g_head.load_state_dict(sa_ckpt["g_head"])
    logger.info("Loaded Stage A checkpoint: %s", stage_a_checkpoint)

    cue_extractor = CueExtractor(cache_dir=cfg.paths.cache, context_encoder_type=ctx_encoder_type)

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

    logger.info(
        "LLM-1 Stage B: %d epochs, LoRA rank=%d, early_stopping_patience=%d",
        n_epochs, lora_cfg.rank, patience,
    )

    n_steps_per_epoch = len(train_loader)
    log_every = max(1, n_steps_per_epoch // 10)

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

            if (step + 1) % log_every == 0 or (step + 1) == n_steps_per_epoch:
                avg_so_far = epoch_loss / (step + 1)
                logger.info(
                    "  [B] Epoch %d/%d | step %d/%d | loss=%.4f",
                    epoch + 1, n_epochs, step + 1, n_steps_per_epoch, avg_so_far,
                )

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
    batch, fusion, classifier, adapter, g_head: "GHeadPerModality", llm, tokenizer,
    cue_extractor, device, lambda_kl, lambda_rec, kl_temp,
    modality_dropout_p=0.0,
):
    """Compute L = L_LM + λ_kl * L_kl + λ_rec * L_rec (per-modality heads)."""
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
        audio, face, context, text_feat, has_face = _modality_dropout(
            audio.clone(), face.clone(), context.clone(), text_feat.clone(),
            p=modality_dropout_p, has_face=has_face.clone(),
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

        transcript = transcripts[i] if transcripts[i] else ""
        if transcript:
            prompt = (
                f'Lời nói: "{transcript}"\n'
                "Dựa trên đặc trưng đa phương thức và lời nói trên, "
                "mô tả các đặc điểm quan sát được và xác định cảm xúc."
            )
        else:
            prompt = "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được và xác định cảm xúc."

        prompts.append(prompt)
        targets.append(target)
        attr_vecs.append(attr_vec)

    attr_tensor = torch.stack(attr_vecs).to(device)

    # --- Soft tokens (text excluded — transcript passed as raw text in prompt) ---
    soft_tokens, soft_mask = adapter(
        fused, penult=penult, audio=audio, face=face,
        context=context, has_face=has_face,
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
    # Cast soft_tokens to match LLM dtype (e.g. bfloat16 for Qwen2.5 with BnB).
    # Adapter stays in float32 for gradient precision; only the LLM input is cast.
    inputs_embeds = torch.cat([soft_tokens.to(full_embeds.dtype), full_embeds], dim=1)
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

    # --- L_rec: per-modality g_head reconstruction loss (tap A only) ---
    if lambda_rec > 0:
        face_pred, voice_pred, motion_pred, text_pred = g_head(
            face, audio, context, text_feat
        )
        # Split attr_tensor into per-modality targets (must match CueExtractor layout)
        face_attrs = attr_tensor[:, :5]
        voice_attrs = attr_tensor[:, 5:8]
        if g_head.has_context:
            motion_attrs = attr_tensor[:, 8:11]
            text_attrs = attr_tensor[:, 11:15]
        else:
            text_attrs = attr_tensor[:, 8:12]

        # Face L_rec: only for samples where a face was actually detected.
        # Clips without webcam have face_attrs=[0]*5, which would pull face_head
        # toward zero and cause collapse. Mask them out entirely.
        face_mask = has_face.float()  # (B,) 1=has_face 0=no_face
        if face_mask.sum() > 0:
            fp_masked = face_pred[face_mask.bool()]
            fa_masked = face_attrs[face_mask.bool()]
            loss_face = F.smooth_l1_loss(fp_masked, fa_masked)
        else:
            loss_face = face_pred.sum() * 0.0  # no-op but keeps graph

        loss_rec = (
            loss_face
            + F.smooth_l1_loss(voice_pred, voice_attrs)
            + F.smooth_l1_loss(text_pred, text_attrs)
        )
        if g_head.has_context and motion_pred is not None:
            loss_rec = loss_rec + F.smooth_l1_loss(motion_pred, motion_attrs)
        total_loss = total_loss + lambda_rec * loss_rec

    # --- L_kl: soft distillation from MLP confidence ---
    # KL(mlp_teacher || llm_student): lm_soft=log_softmax(llm/T), mlp_soft=softmax(mlp/T)
    # F.kl_div(log_Q, P) = KL(P||Q) — direction: MLP→LLM ✓; T² restores gradient magnitude
    # Both distributions are 8-d (same support): mlp_soft=(B,8), lm_soft=log_softmax over 8 tokens (FIX 8.4)
    if lambda_kl > 0:
        mlp_soft = F.softmax(logits / kl_temp, dim=-1).detach()  # (B, 8)
        lm_logits_at_emotion = _extract_emotion_logprobs(lm_out.logits, lm_labels, tokenizer)  # (B, 8) restricted
        if lm_logits_at_emotion is not None:
            lm_soft = F.log_softmax(lm_logits_at_emotion / kl_temp, dim=-1)  # log-normalized over 8 tokens ✓
            loss_kl = F.kl_div(lm_soft, mlp_soft, reduction="batchmean") * (kl_temp ** 2)
            total_loss = total_loss + lambda_kl * loss_kl

    return total_loss


def _build_label_token_ids(tokenizer, label_names: list) -> list:
    """Map each emotion label to a single-token ID (hard requirement).

    All label names MUST tokenize to exactly one token with the current
    tokenizer.  A multi-token label makes the 8-class distribution ambiguous
    (which token represents the class?), so we raise immediately rather than
    silently approximating with the first subword.  If a label is multi-token,
    map it to a single-token surface string before calling (e.g. rename the
    display label or add a one-word alias).

    Also raises if any two labels share the same token ID — a collision would
    corrupt the restricted 8-d KL support.
    """
    ids = []
    for name in label_names:
        toks = tokenizer.encode(name, add_special_tokens=False)
        if not toks:
            raise ValueError(f"Label '{name}' produced empty token encoding")
        if len(toks) > 1:
            raise ValueError(
                f"Label '{name}' encodes as {len(toks)} tokens {toks}; "
                "all emotion labels must map to a single token for consistent "
                "KL and hedge metrics. Rename the label to a single-token surface form."
            )
        ids.append(toks[0])
    if len(set(ids)) != len(ids):
        dupes = [(n, i) for n, i in zip(label_names, ids) if ids.count(i) > 1]
        raise ValueError(
            f"Duplicate first-token IDs across emotion labels: {dupes}. "
            "KL distillation support would be inconsistent."
        )
    return ids


def _extract_emotion_logprobs(lm_logits, labels, tokenizer):
    """Extract raw logits restricted to the 8 emotion token positions.

    Returns shape (B, 8) — the raw (un-normalized) logit for each of the 8
    emotion token IDs at the last supervised position of each sequence.
    The caller applies log_softmax over these 8 logits, which renormalizes
    the distribution to the same 8-token support as the MLP's softmax (FIX 8.4).

    Returns None if extraction fails (KL loss is silently skipped).
    """
    try:
        label_token_ids = _build_label_token_ids(tokenizer, _LABEL_NAMES)
        n_labels = len(label_token_ids)

        B = lm_logits.shape[0]
        emotion_logits = []
        for i in range(B):
            valid = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                return None
            last_pos = valid[-1].item()
            # Restricts to exactly n_labels token IDs; log_softmax over this
            # vector produces a valid 8-class distribution (sum-to-1) matching MLP.
            token_logits = lm_logits[i, last_pos, label_token_ids]  # (n_labels,)
            assert token_logits.shape == (n_labels,), (
                f"Expected {n_labels} emotion logits, got {token_logits.shape}"
            )
            emotion_logits.append(token_logits)
        result = torch.stack(emotion_logits)  # (B, n_labels)
        assert result.shape == (B, n_labels)
        return result
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
    llm.eval()
    total = 0
    n_agree = 0
    n_format = 0

    for batch in val_loader:
        if total >= n_samples:
            break

        audio = batch["audio"].to(device)
        face = batch["face"].to(device)
        context = batch["context"].to(device)
        text_feat = batch["text"].to(device)
        has_face = batch["has_face"].to(device)
        transcripts = batch["transcript"]
        B = audio.shape[0]

        with torch.no_grad():
            fused = fusion(audio, face, context, text_feat, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]
            logits, penult = classifier(fused, return_penultimate=True)

            soft_tokens, soft_mask = adapter(
                fused, penult=penult, audio=audio, face=face,
                context=context, has_face=has_face,
            )

            for i in range(min(B, n_samples - total)):
                mlp_idx = int(logits[i].argmax().item())
                mlp_label = _LABEL_NAMES[mlp_idx]

                transcript = transcripts[i] if transcripts[i] else ""
                if transcript:
                    prompt = (
                        f'Lời nói: "{transcript}"\n'
                        "Dựa trên đặc trưng đa phương thức và lời nói trên, "
                        "mô tả các đặc điểm quan sát được và xác định cảm xúc."
                    )
                else:
                    prompt = "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được và xác định cảm xúc."

                text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                text_embeds = llm.get_input_embeddings()(text_ids)
                inputs_embeds = torch.cat([soft_tokens[i:i+1].to(text_embeds.dtype), text_embeds], dim=1)
                attn_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)

                out_ids = llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=150,
                    do_sample=False,
                    repetition_penalty=1.0,
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
    llm.train()

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
