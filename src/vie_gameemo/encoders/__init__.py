"""Stage 2 encoders for the 4 modalities.

Modules:
    - audio_whisper: Whisper encoder for speech prosody features
    - face_vit: ViT-FER for streamer face (Path 1 of dual-path)
    - context_vit: ViT-ImageNet for gameplay context (Path 2 of dual-path)
    - text_xlmr: XLM-RoBERTa for transcript

All encoders are FROZEN during training (only fusion + classifier are trained).
Outputs are token sequences of shape (B, T, 768) ready for fusion.
"""
