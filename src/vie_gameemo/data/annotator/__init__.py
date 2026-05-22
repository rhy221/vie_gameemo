"""Multi-agent annotation pipeline (Stage 0, Section 2.3 of spec).

This package implements the annotation workflow that re-annotates each clip
with rich multimodal descriptions, then consolidates them into a reasoning
explanation. Memory-conscious: agents are loaded/unloaded serially to fit
on consumer GPUs.

Pipeline (per clip):
    1. peak_frame: detect frame with max AU intensity (OpenFace)
    2. openface_au: extract Action Unit intensities (Cved)
    3. qwen_vl_agent: describe visual scene/context (Cvod)
    4. qwen_audio_agent: describe audio prosody (Catd)
    5. whisper_asr: transcribe speech (Cls)
    6. consolidator: merge all into structured reasoning (Cmd)

Each agent is a class with a `.batch_describe()` method; the orchestrator
in pipeline.py runs them serially.
"""
