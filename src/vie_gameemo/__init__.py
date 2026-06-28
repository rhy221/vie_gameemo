"""Vie-GameEmo: Multimodal Emotion Recognition for Vietnamese Game Livestreams.

A research pipeline combining audio (spectrogram), visual (dual-path: face crop
+ gameplay context), and text (transcript) modalities for emotion classification
of Vietnamese game streamers/reviewers, with LLM-based reasoning explanations.

Architecture:
    - Stage 0: Data curation + multi-agent annotation (runs independently)
    - Stage 1: Demuxing (audio + frames extraction)
    - Stage 2: Encoders (a: Whisper audio, b: dual-path visual, c: XLM-R text)
    - Stage 3: Conv-Attention 4-modality fusion
    - Stage 4: MLP emotion classifier
    - Stage 5: LLM reasoner (4 setups for comparison)

See PROMPT_FOR_CLAUDE_CODE.md and pipeline_implementation_detailed.md for full
design specification.
"""

__version__ = "0.1.0"
