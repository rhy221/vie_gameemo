"""Stage 5 LLM-4: RLVR training (R1-Omni-inspired, Section 9.5 of spec).

Two phases:
    --phase cold-start: SFT on multi-agent reasoning data (50-100 samples)
    --phase rlvr:       GRPO with reward = R_acc + R_format

Run cold-start first, then RLVR with --resume-from on the cold-start checkpoint.

Usage:
    # Phase 1: cold start (brief SFT to learn output format)
    python scripts/train_rlvr.py --config config.yaml --phase cold-start

    # Phase 2: RLVR (long, heavy compute — A100 recommended)
    python scripts/train_rlvr.py --config config.yaml --phase rlvr \\
        --resume-from outputs/checkpoints/llm4_coldstart
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import setup_logging
from vie_gameemo.utils.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLM-4 with RLVR (GRPO)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--phase", choices=["cold-start", "rlvr"], required=True)
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Cold-start adapter dir for RLVR phase")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override base model (e.g., Qwen/Qwen2.5-0.5B-Instruct for debug)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for fast generation in GRPO")
    parser.add_argument("--annotations-dir", type=Path, default=None,
                        help="Override annotations directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)

    if args.phase == "cold-start":
        ckpt = _run_cold_start(cfg, args)
    else:
        if not args.resume_from:
            raise ValueError("--resume-from required for RLVR phase (cold-start checkpoint dir)")
        ckpt = _run_rlvr(cfg, args)

    logger.info("Training complete. Checkpoint: %s", ckpt)
    return 0


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_reasoning_dataset(annotations_dir: Path, split: str = "train") -> list[dict[str, Any]]:
    """Load reasoning examples from annotation JSON files.

    Each annotation must have a 'reasoning' field (multi-agent generated) and
    an 'emotion_label' field. Returns list of dicts with 'prompt' and 'label'.

    Args:
        annotations_dir: Directory with *.json annotation files.
        split: 'train' | 'val' | 'test'. Filters if annotation has 'split' key.

    Returns:
        List of {'prompt': str, 'label': str, 'reasoning': str, 'text': str}.
    """
    from vie_gameemo.llm.llm2_coreasoner import LLM2CoReasoner

    examples = []
    for json_path in sorted(annotations_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", json_path, exc)
            continue

        if data.get("split", split) != split:
            continue

        reasoning = data.get("reasoning", "")
        label = data.get("emotion_label", "")
        if not reasoning or not label:
            continue

        evidence = {
            "face_aus": data.get("face_aus", "N/A"),
            "visual_objective": data.get("visual_objective", "N/A"),
            "audio_tone": data.get("audio_tone", "N/A"),
            "transcript": data.get("transcript", ""),
        }
        prompt = LLM2CoReasoner.build_prompt(evidence)
        target = f"<think>\n{reasoning}\n</think>\n<answer>{label}</answer>"

        examples.append({
            "prompt": prompt,
            "label": label,
            "reasoning": reasoning,
            "text": prompt + "\n" + target,
        })

    logger.info("Loaded %d reasoning examples (split=%s) from %s", len(examples), split, annotations_dir)
    return examples


# ---------------------------------------------------------------------------
# Phase 1: Cold start (SFT)
# ---------------------------------------------------------------------------

def _run_cold_start(cfg, args) -> Path:
    """Phase 1: SFT on multi-agent reasoning data.

    Trains Qwen2.5 with LoRA on annotations that have multi-agent generated
    reasoning (<think>/<answer> targets). Uses HuggingFace SFTTrainer.

    Args:
        cfg: Config namespace.
        args: Parsed CLI args.

    Returns:
        Path to saved adapter directory.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    from vie_gameemo.llm.llm1_explainer import _make_bnb_config

    llm_cfg = cfg.llm
    rlvr_cfg = getattr(cfg.training, "rlvr", None)
    cold_cfg = getattr(rlvr_cfg, "cold_start", None) if rlvr_cfg else None

    base_model = args.base_model or llm_cfg.base_model.name
    n_epochs = args.epochs or (getattr(cold_cfg, "epochs", 3) if cold_cfg else 3)
    lr = getattr(cold_cfg, "learning_rate", 2e-4) if cold_cfg else 2e-4
    batch_size = getattr(cold_cfg, "batch_size", 4) if cold_cfg else 4
    max_seq_len = getattr(cold_cfg, "max_seq_len", 512) if cold_cfg else 512

    lora_rank = getattr(llm_cfg, "lora_rank", 16)
    lora_alpha = getattr(llm_cfg, "lora_alpha", 32)
    lora_dropout = getattr(llm_cfg, "lora_dropout", 0.05)

    output_dir = Path(cfg.paths.checkpoints) / "llm4_coldstart"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = args.annotations_dir or Path(cfg.paths.annotations)

    logger.info("Cold start SFT: model=%s, epochs=%d, lr=%s", base_model, n_epochs, lr)

    examples = _load_reasoning_dataset(annotations_dir, split="train")
    if not examples:
        raise RuntimeError(
            f"No reasoning examples found in {annotations_dir}. "
            "Run stage0_annotate.py first."
        )

    dataset = Dataset.from_list([{"text": e["text"]} for e in examples])

    quant_cfg = _make_bnb_config(llm_cfg.base_model.quantization)
    model_kwargs: dict = {"device_map": "auto"}
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    else:
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Cold start complete. Adapter saved to %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Phase 2: GRPO RLVR
# ---------------------------------------------------------------------------

def _run_rlvr(cfg, args) -> Path:
    """Phase 2: GRPO RLVR training.

    Loads cold-start adapter, then trains with GRPOTrainer using R_acc + R_format
    rewards. Optionally uses vLLM for fast generation rollouts.

    Args:
        cfg: Config namespace.
        args: Parsed CLI args (args.resume_from = cold-start adapter dir).

    Returns:
        Path to final RLVR adapter directory.
    """
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from vie_gameemo.llm.llm1_explainer import _make_bnb_config
    from vie_gameemo.llm.llm4_rlvr import reward_total

    llm_cfg = cfg.llm
    rlvr_cfg = getattr(cfg.training, "rlvr", None)
    grpo_cfg = getattr(rlvr_cfg, "grpo", None) if rlvr_cfg else None

    base_model = args.base_model or llm_cfg.base_model.name
    n_epochs = args.epochs or (getattr(grpo_cfg, "epochs", 1) if grpo_cfg else 1)
    lr = getattr(grpo_cfg, "learning_rate", 5e-6) if grpo_cfg else 5e-6
    batch_size = getattr(grpo_cfg, "batch_size", 2) if grpo_cfg else 2
    n_generations = getattr(grpo_cfg, "num_generations", 4) if grpo_cfg else 4
    max_prompt_len = getattr(grpo_cfg, "max_prompt_len", 256) if grpo_cfg else 256
    max_completion_len = getattr(grpo_cfg, "max_completion_len", 400) if grpo_cfg else 400

    output_dir = Path(cfg.paths.checkpoints) / "llm4_rlvr"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = args.annotations_dir or Path(cfg.paths.annotations)

    logger.info("RLVR GRPO: model=%s, epochs=%d, lr=%s, n_gen=%d",
                base_model, n_epochs, lr, n_generations)

    examples = _load_reasoning_dataset(annotations_dir, split="train")
    if not examples:
        raise RuntimeError(
            f"No reasoning examples found in {annotations_dir}. "
            "Run stage0_annotate.py first."
        )

    dataset = Dataset.from_list([
        {"prompt": e["prompt"], "label": e["label"]}
        for e in examples
    ])

    quant_cfg = _make_bnb_config(llm_cfg.base_model.quantization)
    model_kwargs: dict = {"device_map": "auto"}
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    else:
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    cold_start_dir = str(args.resume_from)
    if Path(cold_start_dir).exists():
        model = PeftModel.from_pretrained(model, cold_start_dir)
        logger.info("Loaded cold-start adapter from %s", cold_start_dir)

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """Closure over ground-truth labels from dataset batch."""
        ground_truths = kwargs.get("label", ["neutral"] * len(completions))
        return reward_total(completions, list(ground_truths))

    grpo_training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        num_generations=n_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        use_vllm=args.use_vllm,
        dataloader_pin_memory=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("RLVR training complete. Adapter saved to %s", output_dir)
    return output_dir


if __name__ == "__main__":
    sys.exit(main())
