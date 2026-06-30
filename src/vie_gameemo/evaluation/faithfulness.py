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
    """Run all 5 faithfulness evaluations.

    Returns dict with agreement, ablation, nn_decode, counterfactual, and hedge results.
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

    # 4. Counterfactual consistency
    counterfactual = eval_counterfactual(fusion, classifier, adapter, llm, tokenizer, val_loader, device, n_samples=min(20, n_samples))
    results.update(counterfactual)

    # 5. Hedge when MLP wrong
    hedge = eval_hedge(fusion, classifier, adapter, llm, tokenizer, val_loader, device, n_samples=n_samples)
    results.update(hedge)

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


def eval_counterfactual(
    fusion, classifier, adapter, llm, tokenizer,
    val_loader, device, n_samples=20,
) -> dict:
    """Counterfactual consistency: perturb one input modality → check explanation changes.

    For each sample: if zeroing a modality causes MLP to change its label,
    the explanation (Cues + Emotion) must change consistently — not stay frozen.

    Metric: counterfactual_consistency_rate = n_consistent / n_label_changed
      - "consistent": LLM Emotion changes to match new MLP label
      - "stale":      LLM Emotion stays on original label (faithfulness failure)
    """
    prompt = (
        "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được "
        "và xác định cảm xúc."
    )

    adapter.eval()
    n_label_changed = 0
    n_consistent = 0
    total = 0

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
            fused_orig = fusion(audio, face, context, text_feat, has_face=has_face)
            if isinstance(fused_orig, tuple):
                fused_orig = fused_orig[0]
            logits_orig, penult_orig = classifier(fused_orig, return_penultimate=True)

            for i in range(min(B, n_samples - total)):
                orig_label = _LABEL_NAMES[int(logits_orig[i].argmax().item())]

                # Try zeroing each modality; use first one that changes MLP label
                modality_sets = [
                    (torch.zeros_like(audio[i:i+1]), face[i:i+1], context[i:i+1], text_feat[i:i+1]),
                    (audio[i:i+1], torch.zeros_like(face[i:i+1]), context[i:i+1], text_feat[i:i+1]),
                    (audio[i:i+1], face[i:i+1], torch.zeros_like(context[i:i+1]), text_feat[i:i+1]),
                    (audio[i:i+1], face[i:i+1], context[i:i+1], torch.zeros_like(text_feat[i:i+1])),
                ]

                for a_p, f_p, c_p, t_p in modality_sets:
                    fused_p = fusion(a_p, f_p, c_p, t_p, has_face=has_face[i:i+1])
                    if isinstance(fused_p, tuple):
                        fused_p = fused_p[0]
                    logits_p, penult_p = classifier(fused_p, return_penultimate=True)
                    new_label = _LABEL_NAMES[int(logits_p[0].argmax().item())]

                    if new_label != orig_label:
                        n_label_changed += 1

                        # Generate explanation for original and perturbed
                        soft_orig, _ = adapter(
                            fused_orig[i:i+1], penult=penult_orig[i:i+1],
                            audio=audio[i:i+1], face=face[i:i+1],
                            context=context[i:i+1], text=text_feat[i:i+1],
                            has_face=has_face[i:i+1],
                        )
                        soft_p, _ = adapter(
                            fused_p, penult=penult_p,
                            audio=a_p, face=f_p, context=c_p, text=t_p,
                            has_face=has_face[i:i+1],
                        )

                        raw_orig = _generate(llm, tokenizer, soft_orig, prompt, device)
                        raw_p = _generate(llm, tokenizer, soft_p, prompt, device)

                        emotion_orig_llm = _extract_emotion(raw_orig)
                        emotion_p_llm = _extract_emotion(raw_p)

                        # Consistent: LLM Emotion on perturbed input matches new MLP label
                        if emotion_p_llm == new_label:
                            n_consistent += 1

                        break  # Only use first modality that changed the label

                total += 1

    consistency_rate = n_consistent / max(1, n_label_changed)
    return {
        "counterfactual_n_label_changed": n_label_changed,
        "counterfactual_n_samples": total,
        "counterfactual_consistency_rate": consistency_rate,
    }


def eval_hedge(
    fusion, classifier, adapter, llm, tokenizer,
    val_loader, device, n_samples=50, hedge_threshold: float = 0.6,
) -> dict:
    """Hedge evaluation: does LLM confidence correlate with MLP confidence?

    Primary metric: Pearson + Spearman correlation between MLP max-softmax
    confidence and LLM Emotion-token probability across all evaluated samples.
    A faithful explainer should express higher confidence when the MLP is
    confident, and lower confidence when the MLP is uncertain.

    Secondary metric: hedge_rate = fraction of MLP-wrong samples where the
    LLM emotion-token probability falls below hedge_threshold.

    Args:
        hedge_threshold: LLM emotion-token prob below this → "hedged".
            Set via evaluation.faithfulness.hedge_threshold in config.yaml.
    """
    prompt = (
        "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được "
        "và xác định cảm xúc."
    )

    # Build and validate label token IDs upfront (FIX 7.2: uniqueness assertion).
    from vie_gameemo.training.llm1_explanation import _build_label_token_ids
    try:
        label_token_ids: list[int] = _build_label_token_ids(tokenizer, _LABEL_NAMES)
    except (ValueError, Exception) as exc:
        logger.warning("eval_hedge: label token ID build failed — %s; skipping", exc)
        return {}

    adapter.eval()
    mlp_confs: list[float] = []
    llm_confs: list[float] = []
    n_mlp_wrong = 0
    n_hedged = 0
    total = 0

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

            mlp_conf_batch = F.softmax(logits, dim=-1).max(dim=-1).values  # (B,)

            for i in range(min(B, n_samples - total)):
                mlp_idx = int(logits[i].argmax().item())
                gt_idx = int(gt_labels[i].item())
                mlp_conf_i = float(mlp_conf_batch[i].item())

                # Generate with scores to capture per-step token probabilities
                text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                text_embeds = llm.get_input_embeddings()(text_ids)
                inputs_embeds = torch.cat([soft_tokens[i:i+1], text_embeds], dim=1)

                out = llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                # Find the generation step that emitted a label token → record prob
                llm_conf_i = 0.0
                if out.scores:
                    for step_logits in out.scores:
                        top_id = int(step_logits[0].argmax().item())
                        if top_id in label_token_ids:
                            llm_conf_i = float(F.softmax(step_logits[0], dim=-1)[top_id].item())
                            break

                mlp_confs.append(mlp_conf_i)
                llm_confs.append(llm_conf_i)

                # Secondary: how often does LLM "hedge" when MLP is wrong?
                if mlp_idx != gt_idx:
                    n_mlp_wrong += 1
                    if llm_conf_i < hedge_threshold:
                        n_hedged += 1

                total += 1

    # Spread guard (FIX 7.5): correlation is meaningless when MLP confidence has
    # near-zero variance (overconfident / saturated MLP).
    import numpy as _np
    mlp_conf_arr = _np.array(mlp_confs) if mlp_confs else _np.array([0.0])
    mlp_conf_std = float(_np.std(mlp_conf_arr))
    mlp_conf_mean = float(_np.mean(mlp_conf_arr))
    _LOW_SPREAD = 0.05
    correlation_reliable = mlp_conf_std >= _LOW_SPREAD
    if not correlation_reliable:
        logger.warning(
            "eval_hedge: mlp_conf spread too low (std=%.4f mean=%.4f) — "
            "Pearson/Spearman unreliable; MLP may be overconfident or uncalibrated.",
            mlp_conf_std, mlp_conf_mean,
        )

    # Primary: confidence correlation (requires scipy; degrade gracefully if absent)
    pearson_r, pearson_p, spearman_r, spearman_p = 0.0, 1.0, 0.0, 1.0
    if len(mlp_confs) >= 2:
        try:
            from scipy.stats import pearsonr, spearmanr
            pearson_r, pearson_p = pearsonr(mlp_confs, llm_confs)
            spearman_r, spearman_p = spearmanr(mlp_confs, llm_confs)
        except ImportError:
            logger.warning("scipy not installed — hedge correlation metrics unavailable")

    return {
        "hedge_confidence_pearson_r": float(pearson_r),
        "hedge_confidence_pearson_p": float(pearson_p),
        "hedge_confidence_spearman_r": float(spearman_r),
        "hedge_confidence_spearman_p": float(spearman_p),
        "hedge_n_mlp_wrong": n_mlp_wrong,
        "hedge_n_hedged": n_hedged,
        "hedge_rate": n_hedged / max(1, n_mlp_wrong),
        "hedge_n_samples": total,
        "hedge_mlp_conf_mean": mlp_conf_mean,
        "hedge_mlp_conf_std": mlp_conf_std,
        "hedge_correlation_reliable": correlation_reliable,
    }
