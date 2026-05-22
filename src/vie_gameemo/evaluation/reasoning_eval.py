"""Reasoning evaluation for LLM outputs (EMER-style).

Scores LLM reasoning quality on:
    - Clue overlap: how well predicted reasoning identifies the same
      multimodal evidence as the ground-truth reasoning.
    - Label overlap: how well the final emotion label matches.
    - Format compliance: % of outputs matching <think>/<answer> format.

Uses a judge LLM (Qwen2.5-7B-Instruct by default) to score clue/label
overlap on a 0-10 scale. 100% open-source — no GPT-4 calls.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
Bạn là người chấm điểm chất lượng lý luận cảm xúc. Đánh giá hai đoạn lý luận sau:

Lý luận chuẩn (ground truth):
{gt_reasoning}

Lý luận dự đoán:
{predicted_reasoning}

Nhãn chuẩn: {gt_label}
Nhãn dự đoán: {predicted_label}

Hãy chấm điểm theo thang 0-10:
1. Clue overlap (bằng chứng đa phương thức): bao nhiêu bằng chứng từ ground truth được đề cập?
2. Label accuracy: nhãn dự đoán có khớp với nhãn chuẩn không? (10 nếu đúng, 0 nếu sai)

Trả lời theo format:
<clue_score>[0-10]</clue_score>
<label_score>[0-10]</label_score>
"""

_FORMAT_PATTERN = re.compile(
    r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL
)


def evaluate_reasoning(
    predictions: list[dict],
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
    quantization: str = "4bit",
) -> dict:
    """Score reasoning quality of LLM outputs.

    Args:
        predictions: List of dicts with keys:
            'predicted_reasoning', 'predicted_label',
            'gt_reasoning', 'gt_label'.
        judge_model: HF model ID for judge.
        quantization: Quantization for judge.

    Returns:
        Dict with keys:
            - clue_overlap_mean: float (0-10)
            - label_overlap_mean: float (0-10)
            - format_compliance: float (0-1)
            - per_sample: list of per-sample score dicts
    """
    fmt_compliance = format_compliance(predictions)

    judge = _load_judge(judge_model, quantization)
    per_sample = []

    for pred in predictions:
        prompt = _JUDGE_PROMPT.format(
            gt_reasoning=pred.get("gt_reasoning", ""),
            predicted_reasoning=pred.get("predicted_reasoning", ""),
            gt_label=pred.get("gt_label", ""),
            predicted_label=pred.get("predicted_label", ""),
        )
        response = _generate_judge(judge, prompt)
        clue_score, label_score = _parse_judge_scores(response, fallback_label_match=(
            pred.get("predicted_label", "").strip().lower() ==
            pred.get("gt_label", "").strip().lower()
        ))
        per_sample.append({
            "clue_score": clue_score,
            "label_score": label_score,
            "predicted_label": pred.get("predicted_label", ""),
            "gt_label": pred.get("gt_label", ""),
        })

    _unload_judge(judge)

    clue_mean = sum(s["clue_score"] for s in per_sample) / max(1, len(per_sample))
    label_mean = sum(s["label_score"] for s in per_sample) / max(1, len(per_sample))

    return {
        "clue_overlap_mean": clue_mean,
        "label_overlap_mean": label_mean,
        "format_compliance": fmt_compliance,
        "per_sample": per_sample,
    }


def format_compliance(predictions: list[dict]) -> float:
    """Compute fraction of predictions matching the <think>/<answer> format.

    Args:
        predictions: List of dicts. Checks 'raw' key if present, else
            reconstructs from 'predicted_reasoning' + 'predicted_label'.

    Returns:
        Fraction in [0, 1].
    """
    if not predictions:
        return 0.0
    n_valid = 0
    for pred in predictions:
        raw = pred.get("raw", "")
        if not raw:
            raw = (
                f"<think>{pred.get('predicted_reasoning', '')}</think>"
                f"<answer>{pred.get('predicted_label', '')}</answer>"
            )
        if _FORMAT_PATTERN.search(raw):
            n_valid += 1
    return n_valid / len(predictions)


def _load_judge(model_name: str, quantization: str) -> dict:
    """Load judge LLM, return state dict."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vie_gameemo.llm.llm1_explainer import _make_bnb_config

    logger.info("Loading judge LLM: %s", model_name)
    bnb_cfg = _make_bnb_config(quantization)
    kwargs: dict = {"device_map": "auto"}
    if bnb_cfg is not None:
        kwargs["quantization_config"] = bnb_cfg
    else:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


def _unload_judge(judge: dict) -> None:
    import gc
    import torch
    judge["model"] = None
    judge["tokenizer"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_judge(judge: dict, prompt: str, max_new_tokens: int = 128) -> str:
    import torch

    tokenizer = judge["tokenizer"]
    model = judge["model"]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    out_ids = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def _parse_judge_scores(response: str, fallback_label_match: bool = False) -> tuple[float, float]:
    """Parse <clue_score> and <label_score> from judge response."""
    clue_match = re.search(r"<clue_score>\s*(\d+(?:\.\d+)?)\s*</clue_score>", response)
    label_match = re.search(r"<label_score>\s*(\d+(?:\.\d+)?)\s*</label_score>", response)

    clue_score = float(clue_match.group(1)) if clue_match else 5.0
    label_score = float(label_match.group(1)) if label_match else (10.0 if fallback_label_match else 0.0)

    return min(10.0, max(0.0, clue_score)), min(10.0, max(0.0, label_score))
