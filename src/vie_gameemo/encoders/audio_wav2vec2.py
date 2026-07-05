"""Audio encoder using Wav2Vec2 (self-supervised speech representations).

Wav2Vec2 is pretrained via contrastive self-supervision directly on raw
waveforms (no ASR fine-tuning by default), which tends to preserve
paralinguistic/prosodic detail that ASR-fine-tuned checkpoints can wash out.
Included as an ablation alternative to the Whisper encoder.

Model variants and hidden sizes:
    - facebook/wav2vec2-base:          768d (95M)  <- default
    - facebook/wav2vec2-large:         1024d (317M)
    - facebook/wav2vec2-large-xlsr-53: 1024d (317M, multilingual)

Output: token sequence (B, target_tokens, d_out), adaptively pooled. d_out
defaults to the model's native hidden size (self.d_out); see audio_whisper.py
docstring for why dimension standardization is left to `fusion.audio_dim`
instead of a projection inside this frozen, no_grad encoder.
"""

import logging
from pathlib import Path

import librosa
import torch
from torch import Tensor, nn
from transformers import AutoFeatureExtractor, Wav2Vec2Model

logger = logging.getLogger(__name__)


class Wav2Vec2AudioEncoder(nn.Module):
    """Wav2Vec2 encoder wrapper for emotion-relevant audio features."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        target_tokens: int = 64,
        d_out: int | None = None,
        sample_rate: int = 16000,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize Wav2Vec2 encoder.

        Args:
            model_name: HuggingFace Wav2Vec2 model ID.
            target_tokens: Output sequence length after adaptive pooling.
            d_out: If set and different from the model's native hidden size,
                adds an (untrained, frozen-context) nn.Linear projection.
                Leave as None (default) to output the native hidden size and
                let `fusion.audio_dim` handle standardization instead.
            sample_rate: Input audio sample rate (must be 16kHz for Wav2Vec2).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.target_tokens = target_tokens
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        logger.info("Loading Wav2Vec2 model: %s", model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
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
        logger.info("Wav2Vec2 encoder loaded and frozen (d_model=%d, d_out=%d)", d_model, self.d_out)

    @torch.no_grad()
    def encode(self, audio_path: Path) -> Tensor:
        """Encode a single audio file.

        Args:
            audio_path: Path to wav file (16kHz, mono).

        Returns:
            Tensor of shape (1, target_tokens, self.d_out).

        Raises:
            FileNotFoundError: If audio_path missing.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
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
