"""Audio encoder using HuBERT (self-supervised speech representations).

HuBERT is pretrained via masked prediction of clustered acoustic units on raw
waveforms (no ASR fine-tuning by default). Like Wav2Vec2, it tends to retain
prosodic/paralinguistic detail. Included as an ablation alternative to the
Whisper encoder.

Model variants and hidden sizes:
    - facebook/hubert-base-ls960:   768d (95M)   <- default
    - facebook/hubert-large-ll60k:  1024d (317M)
    - facebook/hubert-xlarge-ll60k: 1280d (964M)

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
from transformers import AutoFeatureExtractor, HubertModel

logger = logging.getLogger(__name__)


class HubertAudioEncoder(nn.Module):
    """HuBERT encoder wrapper for emotion-relevant audio features."""

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        target_tokens: int = 64,
        d_out: int | None = None,
        sample_rate: int = 16000,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize HuBERT encoder.

        Args:
            model_name: HuggingFace HuBERT model ID.
            target_tokens: Output sequence length after adaptive pooling.
            d_out: If set and different from the model's native hidden size,
                adds an (untrained, frozen-context) nn.Linear projection.
                Leave as None (default) to output the native hidden size and
                let `fusion.audio_dim` handle standardization instead.
            sample_rate: Input audio sample rate (must be 16kHz for HuBERT).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.target_tokens = target_tokens
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        logger.info("Loading HuBERT model: %s", model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
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
        logger.info("HuBERT encoder loaded and frozen (d_model=%d, d_out=%d)", d_model, self.d_out)

    @torch.no_grad()
    def encode(self, audio: Path | np.ndarray) -> Tensor:
        """Encode a single audio file or pre-loaded waveform.

        Args:
            audio: Path to wav file (16kHz, mono), or a pre-loaded/augmented
                1D float32 waveform already at `self.sample_rate` (e.g. from
                `augment.audio_augment.augment_waveform`).

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
        input_values = inputs.input_values.to(self.device)

        hidden = self.model(input_values).last_hidden_state  # (1, N, d_model)

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
        tensors = [self.encode(p) for p in audio_paths]
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _adaptive_pool(x: Tensor, target_len: int) -> Tensor:
        """Adaptive average pool along sequence dim to target_len.

        Args:
            x: Tensor of shape (B, T, D).
            target_len: Target sequence length.

        Returns:
            Pooled tensor (B, target_len, D).
        """
        x_t = x.transpose(1, 2)
        pooled = nn.functional.adaptive_avg_pool1d(x_t, target_len)
        return pooled.transpose(1, 2)
