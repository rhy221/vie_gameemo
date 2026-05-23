"""LLM-4: RLVR-trained reasoner (R1-Omni-inspired, Section 9.5 of spec).

Reinforcement Learning with Verifiable Reward + Group Relative Policy
Optimization (GRPO). Trains the LLM to produce structured reasoning that
both (a) predicts the correct emotion label, and (b) follows the required
output format.

Pipeline:
    1. Cold start: brief SFT on multi-agent reasoning data (50-100 samples).
    2. RLVR: GRPO with rewards:
        R_acc:     1 if predicted answer == ground truth, else 0
        R_format:  1 if output matches <think>...</think><answer>...</answer>
        R = R_acc + R_format

Expected benefit (per R1-Omni paper): +10-15% OOD generalization vs SFT-only.
"""

import logging
import re
from pathlib import Path

from vie_gameemo.llm.base import BaseLLMReasoner, LLMOutput
from vie_gameemo.llm.llm1_explainer import _make_bnb_config
from vie_gameemo.llm.llm2_coreasoner import LLM2CoReasoner

logger = logging.getLogger(__name__)

_VALID_LABELS = ["neutral", "focus", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


# ---------------------------------------------------------------------------
# Reward functions (used by GRPOTrainer)
# ---------------------------------------------------------------------------

def reward_accuracy(
    completions: list[str],
    ground_truths: list[str],
) -> list[float]:
    """R_acc: 1.0 if extracted <answer> matches ground truth, else 0.0.

    Args:
        completions: Model generations.
        ground_truths: Per-sample ground-truth emotion labels.

    Returns:
        List of 0/1 floats.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            predicted = match.group(1).strip().lower()
            rewards.append(1.0 if predicted == gt.strip().lower() else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_format(
    completions: list[str],
    pattern: str = r"<think>.*?</think>\s*<answer>.*?</answer>",
) -> list[float]:
    """R_format: 1.0 if completion matches required pattern, else 0.0."""
    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        rewards.append(1.0 if match else 0.0)
    return rewards


def reward_total(
    completions: list[str],
    ground_truths: list[str],
    acc_weight: float = 1.0,
    format_weight: float = 1.0,
) -> list[float]:
    """Combined reward: acc_weight * R_acc + format_weight * R_format."""
    r_acc = reward_accuracy(completions, ground_truths)
    r_fmt = reward_format(completions)
    return [acc_weight * a + format_weight * f for a, f in zip(r_acc, r_fmt)]


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class LLM4RLVR(BaseLLMReasoner):
    """RLVR-trained reasoner (inference-only API; training in train_rlvr.py)."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Path | None = None,
        quantization: str = "4bit",
        max_new_tokens: int = 400,
        temperature: float = 0.7,
        use_vllm: bool = False,
    ) -> None:
        """Initialize for inference.

        Args:
            base_model: HF base model ID.
            adapter_path: Path to RLVR-trained LoRA adapter (must exist).
            quantization: '4bit' | '8bit' | 'none'.
            max_new_tokens: Max gen length.
            temperature: Sampling temp.
            use_vllm: Use vLLM for fast inference (requires vllm package).
        """
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_vllm = use_vllm
        self.model = None
        self.tokenizer = None
        self._vllm_model = None

    def load(self) -> None:
        """Load base model + LoRA adapter for inference."""
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_hf()

    def _load_hf(self) -> None:
        """Load with HuggingFace transformers + optional PEFT adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM-4 base: %s (quant=%s)", self.base_model, self.quantization)
        bnb_cfg = _make_bnb_config(self.quantization)
        kwargs: dict = {"device_map": "auto"}
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        else:
            kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)

        if self.adapter_path is not None and Path(self.adapter_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
            logger.info("Loaded RLVR LoRA adapter from %s", self.adapter_path)

        self.model.eval()
        logger.info("LLM-4 loaded (HF backend)")

    def _load_vllm(self) -> None:
        """Load with vLLM for fast batched inference."""
        try:
            from vllm import LLM as VLLMModel
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM not installed. Install with: pip install vllm") from exc

        logger.info("Loading LLM-4 with vLLM: %s", self.base_model)
        enable_lora = self.adapter_path is not None and Path(self.adapter_path).exists()
        self._vllm_model = VLLMModel(
            model=self.base_model,
            dtype="float16",
            enable_lora=enable_lora,
            max_lora_rank=64,
        )
        self._vllm_sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        logger.info("LLM-4 loaded (vLLM backend, lora=%s)", enable_lora)

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch

        self.model = None
        self.tokenizer = None
        self._vllm_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reason(self, evidence: dict) -> LLMOutput:
        """Generate <think>/<answer> for one sample.

        Args:
            evidence: Same as LLM-2 (modality descriptions + transcript):
                'face_aus', 'visual_objective', 'audio_tone', 'transcript'.
                Optionally 'emotion_categories'.

        Returns:
            LLMOutput.
        """
        if self.model is None and self._vllm_model is None:
            self.load()

        prompt = LLM2CoReasoner.build_prompt(evidence)

        if self.use_vllm and self._vllm_model is not None:
            raw = self._generate_vllm([prompt])[0]
        else:
            raw = self._generate_hf(prompt)

        from vie_gameemo.llm.llm1_explainer import LLM1Explainer
        reasoning, answer, fmt_valid = LLM1Explainer.parse_output(raw)
        return LLMOutput(reasoning=reasoning, answer=answer, raw=raw, format_valid=fmt_valid)

    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        """Batch reason.

        Args:
            evidences: List of evidence dicts.

        Returns:
            List of LLMOutput.
        """
        if self.model is None and self._vllm_model is None:
            self.load()

        from vie_gameemo.llm.llm1_explainer import LLM1Explainer

        prompts = [LLM2CoReasoner.build_prompt(e) for e in evidences]

        if self.use_vllm and self._vllm_model is not None:
            raws = self._generate_vllm(prompts)
        else:
            raws = [self._generate_hf(p) for p in prompts]

        outputs = []
        for raw in raws:
            reasoning, answer, fmt_valid = LLM1Explainer.parse_output(raw)
            outputs.append(LLMOutput(reasoning=reasoning, answer=answer, raw=raw, format_valid=fmt_valid))
        return outputs

    def _generate_hf(self, prompt: str) -> str:
        """Run text generation for a single prompt (HF backend)."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _generate_vllm(self, prompts: list[str]) -> list[str]:
        """Run batched generation with vLLM backend."""
        from vllm.lora.request import LoRARequest

        lora_request = None
        if self.adapter_path is not None and Path(self.adapter_path).exists():
            lora_request = LoRARequest("rlvr_adapter", 1, str(self.adapter_path))

        outputs = self._vllm_model.generate(
            prompts,
            sampling_params=self._vllm_sampling_params,
            lora_request=lora_request,
        )
        return [o.outputs[0].text.strip() for o in outputs]
