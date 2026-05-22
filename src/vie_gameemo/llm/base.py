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

    Implementations: LLM1Explainer, LLM2CoReasoner, LLM3VLMEndToEnd, LLM4RLVR.

    All methods accept a dict of multimodal evidence and return a structured
    LLMOutput.
    """

    @abstractmethod
    def reason(self, evidence: dict) -> LLMOutput:
        """Generate reasoning + emotion prediction.

        Args:
            evidence: Dict with keys depending on setup:
                Common: 'transcript', 'face_aus', 'visual_objective', 'audio_tone'
                LLM-1 also needs: 'label' (predicted by classifier upstream)
                LLM-3 also needs: 'video_frames', 'audio_path'

        Returns:
            LLMOutput with reasoning, answer, raw text, and format flag.
        """
        ...

    @abstractmethod
    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        """Batch version for efficiency."""
        ...
