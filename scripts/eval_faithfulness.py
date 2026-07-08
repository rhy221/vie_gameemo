"""Evaluate LLM-1 Faithful Explainer faithfulness.

Runs 3 evaluations:
  1. Agreement: LLM Emotion vs MLP argmax
  2. Tap A ablation: zero raw tokens → check Cues degrade, Emotion stable
  3. NN-decode: project soft tokens to nearest vocab embeddings

Usage:
    python scripts/eval_faithfulness.py \\
        --config config.yaml \\
        --perception-ckpt outputs/checkpoints/perception_best.pt \\
        --llm1-ckpt outputs/checkpoints/llm1_explanation_best.pt \\
        --split val \\
        --n-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from vie_gameemo.data.dataset import VieGameEmoDataset
from vie_gameemo.evaluation.faithfulness import evaluate_faithfulness
from vie_gameemo.training.llm1_explanation import collate_fn_llm1
from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import get_logger, setup_logging
from vie_gameemo.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM-1 faithfulness")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--perception-ckpt", type=Path, required=True)
    parser.add_argument("--llm1-ckpt", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    device = torch.device(
        cfg.compute.device if cfg.compute.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    from vie_gameemo.classifiers import get_classifier
    from vie_gameemo.fusion import get_fusion
    from vie_gameemo.llm.modal_adapter import ModalAdapter
    from vie_gameemo.training.llm1_explanation import _make_bnb_config

    fcfg = cfg.fusion
    ccfg = cfg.classifier
    llm_cfg = cfg.llm

    # Load frozen perception
    fusion = get_fusion(
        fcfg.type, d_model=fcfg.d_model, n_modalities=fcfg.n_modalities,
        n_conv_blocks=getattr(fcfg, "n_conv_blocks", 4),
        kernel_size=getattr(fcfg, "kernel_size", 3),
        align_to=getattr(fcfg, "align_to", "audio"),
        return_attention=False,
        skip_mlp_if_matched=getattr(fcfg, "skip_mlp_if_matched", False),
    ).to(device)
    p_ckpt = torch.load(args.perception_ckpt, map_location="cpu")
    classifier = get_classifier(
        ccfg, d_model=fcfg.d_model, device=device,
        classifier_type=p_ckpt.get("classifier_type"),
    )
    fusion.load_state_dict(p_ckpt["fusion_state_dict"])
    classifier.load_state_dict(p_ckpt["classifier_state_dict"])
    fusion.eval()
    classifier.eval()

    # Load LLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = getattr(llm_cfg.base_model, "fallback", llm_cfg.base_model.name)
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

    # Load LLM-1 checkpoint
    llm1_ckpt = torch.load(args.llm1_ckpt, map_location="cpu")
    llm_hidden = llm.config.hidden_size
    adapter = ModalAdapter(
        d_fusion=fcfg.d_model, d_llm=llm_hidden,
        d_penult=ccfg.hidden_dim,
    ).to(device)
    adapter.load_state_dict(llm1_ckpt["llm_adapter"], strict=False)
    adapter.eval()

    if llm1_ckpt.get("llm_peft") is not None:
        try:
            from peft import LoraConfig, get_peft_model
            tcfg = cfg.training.llm1_explanation
            lora_config = LoraConfig(
                r=tcfg.lora.rank, lora_alpha=tcfg.lora.alpha,
                target_modules=list(tcfg.lora.target_modules),
                bias="none", task_type="CAUSAL_LM",
            )
            llm = get_peft_model(llm, lora_config)
            llm.load_state_dict(llm1_ckpt["llm_peft"], strict=False)
        except Exception as e:
            logger.warning("Failed to load LoRA: %s", e)
    llm.eval()

    # Dataset
    split_manifest = Path(getattr(cfg.paths, "split_manifest", "data/splits.json"))
    ds = VieGameEmoDataset(
        annotations_dir=Path(cfg.paths.annotations),
        features_dir=Path(cfg.paths.features),
        split=args.split,
        split_manifest=split_manifest if split_manifest.exists() else None,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn_llm1)

    # Evaluate
    logger.info("Running faithfulness evaluation on split=%s, n=%d", args.split, args.n_samples)
    results = evaluate_faithfulness(
        fusion, classifier, adapter, llm, tokenizer,
        loader, device, n_samples=args.n_samples,
    )

    # Report
    logger.info("=" * 60)
    logger.info("FAITHFULNESS RESULTS")
    logger.info("=" * 60)
    logger.info("Agreement (LLM vs MLP):   %.4f", results["agreement"])
    logger.info("Format compliance:        %.4f", results["format_rate"])
    logger.info("MLP accuracy vs gold:     %.4f", results["mlp_accuracy_vs_gold"])
    logger.info("LLM accuracy vs gold:     %.4f", results["llm_accuracy_vs_gold"])
    logger.info("Ablation emotion stable:  %.4f", results["ablation_emotion_stability"])
    logger.info("Ablation cue degradation: %.4f", results["ablation_cue_degradation"])
    logger.info("NN-decode samples:        %d", len(results.get("nn_decode_samples", [])))
    logger.info("=" * 60)

    # Save
    output_path = args.output or Path(cfg.paths.results) / "faithfulness_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {k: v for k, v in results.items() if k != "nn_decode_samples"}
    serializable["nn_decode_samples"] = results.get("nn_decode_samples", [])
    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
