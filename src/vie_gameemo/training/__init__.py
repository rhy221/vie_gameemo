"""Training modules: curriculum (perception + cognition) and losses.

Curriculum (Section 10 of spec):
    - Stage 1 — Perception: train fusion + classifier only (recognition).
    - Stage 2 — Cognition: joint train LLM + adapters (recognition + reasoning).

RLVR training (LLM-4) is in scripts/train_rlvr.py (separate compute profile).
"""
