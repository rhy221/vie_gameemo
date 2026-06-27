"""Abstract base for LLM reasoner setups.

All 4 LLM setups (LLM-1 through LLM-4) implement this interface so that
evaluation and inference code can treat them uniformly.

Output format (for consistency with R1-Omni and ablation):
    <think>
    [reasoning here]
    </think>
    <answer>{emotion_label}</answer>
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMOutput:
    """Structured output from any LLM reasoner setup.

    Attributes:
        reasoning: The <think> content (Vietnamese, 3-5 sentences typical).
        answer: The predicted emotion label.
        raw: Raw model output (for debugging / format compliance check).
        format_valid: True if raw matched the expected <think>/<answer> format.
    """
    reasoning: str
    answer: str
    raw: str
    format_valid: bool


class BaseLLMReasoner(ABC):
    """Abstract base for emotion reasoners.

    Implementations: LLM1Explainer, LLM2CoReasoner, LLM3PureReasoner, LLM4RLVR.
    All use soft tokens from ModalAdapter (fusion embedding → LLM space).
    """

    @abstractmethod
    def reason(self, evidence: dict) -> LLMOutput:
        """Generate reasoning + emotion prediction.

        Args:
            evidence: Dict with keys depending on setup:
                LLM-1: 'label' + 'fusion_emb' (+ optional 'transcript')
                LLM-2: 'mlp_label' + 'fusion_emb' (+ optional 'transcript')
                LLM-3: 'fusion_emb' only
                LLM-4: 'fusion_emb' only

        Returns:
            LLMOutput with reasoning, answer, raw text, and format flag.
        """
        ...

    @abstractmethod
    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        """Batch version for efficiency."""
        ...
