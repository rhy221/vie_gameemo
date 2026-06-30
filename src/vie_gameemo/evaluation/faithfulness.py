"""Faithfulness evaluation for LLM-1 Faithful Explainer.

Three evaluations:
  1. Tap A ablation: zero raw modality tokens → Cues should degrade,
     Emotion should stay stable.
  2. Agreement: LLM Emotion vs MLP argmax — should be high.
  3. NN-decode: project soft tokens to nearest vocab embeddings.
"""

import logging
import re
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_LABEL_NAMES = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


def evaluate_faithfulness(
    fusion,
    classifier,
    adapter,
    llm,
    tokenizer,
    val_loader,
    device: torch.device,
    n_samples: int = 50,
) -> dict:
    """Run all 3 faithfulness evaluations.

    Returns dict with agreement, ablation, and nn_decode results.
    """
    results = {}

    # 1. Agreement
    agreement = eval_agreement(fusion, classifier, adapter, llm, tokenizer, val_loader, device, n_samples)
    results.update(agreement)

    # 2. Tap A ablation
    ablation = eval_tap_a_ablation(fusion, classifier, adapter, llm, tokenizer, val_loader, device, n_samples=min(20, n_samples))
    results.update(ablation)

    # 3. NN-decode
    nn_decode = eval_nn_decode(fusion, classifier, adapter, llm, val_loader, device, n_samples=5)
    results["nn_decode_samples"] = nn_decode

    return results


def eval_agreement(
    fusion, classifier, adapter, llm, tokenizer,
    val_loader, device, n_samples=50,
) -> dict:
    """Compare LLM Emotion output vs MLP argmax."""
    prompt = (
        "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được "
        "và xác định cảm xúc."
    )

    adapter.eval()
    total = 0
    n_agree = 0
    n_format = 0
    gt_correct_mlp = 0
    gt_correct_llm = 0

    for batch in val_loader:
        if total >= n_samples:
            break

        audio = batch["audio"].to(device)
        face = batch["face"].to(device)
        context = batch["context"].to(device)
        text_feat = batch["text"].to(device)
        gt_labels = batch["label"].to(device)
        has_face = batch["has_face"].to(device)
        B = audio.shape[0]

        with torch.no_grad():
            fused = fusion(audio, face, context, text_feat, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]
            logits, penult = classifier(fused, return_penultimate=True)

            soft_tokens, _ = adapter(
                fused, penult=penult, audio=audio, face=face,
                context=context, text=text_feat, has_face=has_face,
            )

            for i in range(min(B, n_samples - total)):
                mlp_idx = int(logits[i].argmax().item())
                mlp_label = _LABEL_NAMES[mlp_idx]
                gt_idx = int(gt_labels[i].item())
                gt_label = _LABEL_NAMES[gt_idx]

                if mlp_idx == gt_idx:
                    gt_correct_mlp += 1

                text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                text_embeds = llm.get_input_embeddings()(text_ids)
                inputs_embeds = torch.cat([soft_tokens[i:i+1], text_embeds], dim=1)

                out_ids = llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=150, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                raw = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

                emotion_match = re.search(r"Emotion:\s*(\w+)", raw)
                if emotion_match:
                    n_format += 1
                    predicted = emotion_match.group(1).strip().lower()
                    if predicted == mlp_label:
                        n_agree += 1
                    if predicted == gt_label:
                        gt_correct_llm += 1

                total += 1

    return {
        "agreement": n_agree / max(1, total),
        "format_rate": n_format / max(1, total),
        "mlp_accuracy_vs_gold": gt_correct_mlp / max(1, total),
        "llm_accuracy_vs_gold": gt_correct_llm / max(1, total),
        "n_samples": total,
    }


def eval_tap_a_ablation(
    fusion, classifier, adapter, llm, tokenizer,
    val_loader, device, n_samples=20,
) -> dict:
    """Zero raw modality tokens → check Cues degrade while Emotion stays."""
    prompt = (
        "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được "
        "và xác định cảm xúc."
    )

    adapter.eval()
    total = 0
    emotion_stable = 0
    cue_changed = 0

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

            for i in range(min(B, n_samples - total)):
                # Normal generation
                soft_normal, _ = adapter(
                    fused[i:i+1], penult=penult[i:i+1],
                    audio=audio[i:i+1], face=face[i:i+1],
                    context=context[i:i+1], text=text_feat[i:i+1],
                    has_face=has_face[i:i+1],
                )
                raw_normal = _generate(llm, tokenizer, soft_normal, prompt, device)

                # Ablated: zero raw tokens (keep penult + fusion)
                zero_audio = torch.zeros_like(audio[i:i+1])
                zero_face = torch.zeros_like(face[i:i+1])
                zero_ctx = torch.zeros_like(context[i:i+1])
                zero_text = torch.zeros_like(text_feat[i:i+1])

                soft_ablated, _ = adapter(
                    fused[i:i+1], penult=penult[i:i+1],
                    audio=zero_audio, face=zero_face,
                    context=zero_ctx, text=zero_text,
                    has_face=has_face[i:i+1],
                )
                raw_ablated = _generate(llm, tokenizer, soft_ablated, prompt, device)

                # Compare
                emotion_normal = _extract_emotion(raw_normal)
                emotion_ablated = _extract_emotion(raw_ablated)
                cues_normal = _extract_cues(raw_normal)
                cues_ablated = _extract_cues(raw_ablated)

                if emotion_normal == emotion_ablated:
                    emotion_stable += 1
                if cues_normal != cues_ablated:
                    cue_changed += 1

                total += 1

    return {
        "ablation_emotion_stability": emotion_stable / max(1, total),
        "ablation_cue_degradation": cue_changed / max(1, total),
        "ablation_n_samples": total,
    }


def eval_nn_decode(
    fusion, classifier, adapter, llm,
    val_loader, device, n_samples=5,
) -> list[dict]:
    """Project soft tokens → nearest vocab embeddings (sanity check)."""
    adapter.eval()
    embed_matrix = llm.get_input_embeddings().weight.detach()
    results = []
    total = 0

    for batch in val_loader:
        if total >= n_samples:
            break

        audio = batch["audio"].to(device)
        face = batch["face"].to(device)
        context = batch["context"].to(device)
        text_feat = batch["text"].to(device)
        has_face = batch["has_face"].to(device)
        clip_ids = batch["clip_id"]

        with torch.no_grad():
            fused = fusion(audio, face, context, text_feat, has_face=has_face)
            if isinstance(fused, tuple):
                fused = fused[0]
            logits, penult = classifier(fused, return_penultimate=True)

            soft_tokens, _ = adapter(
                fused, penult=penult, audio=audio, face=face,
                context=context, text=text_feat, has_face=has_face,
            )

            for i in range(min(soft_tokens.shape[0], n_samples - total)):
                tokens = soft_tokens[i]  # (T, d_llm)
                nn_words = []
                for t_idx in range(min(tokens.shape[0], 10)):
                    tok = tokens[t_idx]
                    sims = F.cosine_similarity(tok.unsqueeze(0), embed_matrix, dim=-1)
                    top3 = sims.topk(3)
                    from transformers import AutoTokenizer
                    words = [llm.config._name_or_path]  # placeholder
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(llm.config._name_or_path)
                        words = [tokenizer.decode([idx.item()]) for idx in top3.indices]
                    except Exception:
                        words = [f"id_{idx.item()}" for idx in top3.indices]
                    nn_words.append({
                        "token_idx": t_idx,
                        "top3": words,
                        "top3_sim": [f"{s:.3f}" for s in top3.values.tolist()],
                    })

                results.append({
                    "clip_id": clip_ids[i],
                    "n_soft_tokens": tokens.shape[0],
                    "nearest_neighbors": nn_words,
                })
                total += 1

    return results


def _generate(llm, tokenizer, soft_tokens, prompt, device) -> str:
    text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    text_embeds = llm.get_input_embeddings()(text_ids)
    inputs_embeds = torch.cat([soft_tokens, text_embeds], dim=1)
    out_ids = llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=150, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def _extract_emotion(raw: str) -> str:
    m = re.search(r"Emotion:\s*(\w+)", raw)
    return m.group(1).strip().lower() if m else ""


def _extract_cues(raw: str) -> str:
    m = re.search(r"Cues:\s*(.*?)(?:\.\s*Emotion:|$)", raw, re.DOTALL)
    return m.group(1).strip() if m else ""
