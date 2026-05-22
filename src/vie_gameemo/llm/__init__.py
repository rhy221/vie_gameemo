"""Stage 5 LLM reasoner modules.

4 setups to compare (Section 9 of spec):
    - LLM-1: post-hoc Explainer (no training, prompt-only)
    - LLM-2: Co-Reasoner (modality-to-text + LLM aggregation)
    - LLM-3: VLM end-to-end with LoRA fine-tune
    - LLM-4: RLVR-trained with GRPO (R1-Omni-inspired)

All inherit from `BaseLLMReasoner` to provide a uniform API for evaluation.
"""

from vie_gameemo.llm.base import BaseLLMReasoner

__all__ = ["BaseLLMReasoner"]
