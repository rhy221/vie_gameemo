"""Data layer: schemas, crawler, dataset, feature cache, multi-agent annotation.

Submodules:
    - schemas: Pydantic models for typed data objects (Clip, Annotation, etc.)
    - crawler: Download videos from URL list via yt-dlp
    - dataset: PyTorch Dataset for training/eval
    - feature_cache: Precompute and cache frozen encoder outputs
    - annotator: Multi-agent annotation pipeline (Stage 0)
"""
