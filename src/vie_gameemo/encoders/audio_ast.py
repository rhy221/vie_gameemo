"""Audio encoder using AST (Audio Spectrogram Transformer).

AST is pretrained on AudioSet (audio event classification). It treats the
log-mel spectrogram as an image and applies a ViT-style transformer.
For our use case, AST captures non-speech audio events (laughs, shouts,
sighs) that complement Whisper's speech transcription.

Output: token sequence (B, target_tokens, d_out), adaptively pooled. d_out
defaults to the model's native hidden size (self.d_out); see audio_whisper.py
docstring for why dimension standardization is left to `fusion.audio_dim`
instead of a projection inside this frozen, no_grad encoder.
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor, nn
from transformers import ASTFeatureExtractor, ASTModel

logger = logging.getLogger(__name__)


class ASTAudioEncoder(nn.Module):
    """AST encoder wrapper for emotion-relevant audio features."""

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        target_tokens: int = 64,
        d_out: int | None = None,
        sample_rate: int = 16000,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize AST encoder.

        Args:
            model_name: HuggingFace model ID.
            target_tokens: Output sequence length after adaptive pooling.
            d_out: If set and different from the model's native hidden size,
                adds an (untrained, frozen-context) nn.Linear projection.
                Leave as None (default) to output the native hidden size and
                let `fusion.audio_dim` handle standardization instead.
            sample_rate: Input audio sample rate (must be 16kHz for AST).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.target_tokens = target_tokens
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        logger.info("Loading AST model: %s", model_name)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)

        d_model = self.model.config.hidden_size
        if d_out is not None and d_model != d_out:
            self.proj = nn.Linear(d_model, d_out)
            self.proj = self.proj.to(self.device)
            logger.warning(
                "Added UNTRAINED projection %d -> %d inside frozen encoder "
                "(runs under @torch.no_grad during feature extraction, never "
                "learned). Prefer d_out=None + fusion.audio_dim=%d so the "
                "fusion model's own trainable MLP standardizes the dim.",
                d_model, d_out, d_model,
            )
        else:
            self.proj = None

        self.d_out = d_out if self.proj is not None else d_model
        logger.info("AST encoder loaded and frozen (d_model=%d, d_out=%d)", d_model, self.d_out)

    @torch.no_grad()
    def encode(self, audio: Path | np.ndarray) -> Tensor:
        """Encode a single audio file or pre-loaded waveform.

        Args:
            audio: Path to wav file (sample_rate Hz, mono), or a pre-loaded/
                augmented 1D float32 waveform already at `self.sample_rate`
                (e.g. from `augment.audio_augment.augment_waveform`).

        Returns:
            Tensor of shape (1, target_tokens, self.d_out).

        Raises:
            FileNotFoundError: If a Path is given and it doesn't exist.
        """
        if isinstance(audio, Path):
            if not audio.exists():
                raise FileNotFoundError(f"Audio file not found: {audio}")
            audio, _ = librosa.load(str(audio), sr=self.sample_rate, mono=True)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, N, d_model)
        if self.proj is not None:
            hidden = self.proj(hidden)
        return self._adaptive_pool(hidden, self.target_tokens)

    @torch.no_grad()
    def encode_batch(self, audio_paths: list[Path]) -> Tensor:
        """Batch encode audio files.

        Args:
            audio_paths: List of wav file paths.

        Returns:
            Tensor of shape (N, target_tokens, self.d_out).
        """
        tensors = [self.encode(p) for p in audio_paths]  # each (1, T, d_out)
        return torch.cat(tensors, dim=0)  # (N, T, d_out)

    @staticmethod
    def _adaptive_pool(x: Tensor, target_len: int) -> Tensor:
        """Adaptive average pool along sequence dim to target_len.

        Args:
            x: Tensor of shape (B, T, D).
            target_len: Target sequence length.

        Returns:
            Pooled tensor (B, target_len, D).
        """
        x_t = x.transpose(1, 2)  # (B, D, T)
        pooled = nn.functional.adaptive_avg_pool1d(x_t, target_len)
        return pooled.transpose(1, 2)  # (B, target_len, D)
