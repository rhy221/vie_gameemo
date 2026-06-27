"""Stage 5 LLM reasoner modules.

4 setups — mức độ tự chủ tăng dần, tất cả dùng soft token từ ModalAdapter:
    - LLM-1: Explainer (soft token + nhãn MLP → giải thích, không override)
    - LLM-2: Co-Reasoner (soft token + nhãn MLP hint → có thể override)
    - LLM-3: Pure Reasoner (soft token only → tự dự đoán hoàn toàn)
    - LLM-4: RLVR-trained (soft token only + GRPO reinforcement learning)

All inherit from `BaseLLMReasoner` to provide a uniform API for evaluation.
"""

from vie_gameemo.llm.base import BaseLLMReasoner

__all__ = ["BaseLLMReasoner"]
