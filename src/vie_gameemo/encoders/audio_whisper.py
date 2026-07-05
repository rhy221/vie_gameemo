"""Audio encoder using Whisper encoder (speech prosody features).

Uses only the Whisper encoder (not decoder) to extract speech prosody
representations — intonation, pitch contour, stress, speaking rate, pauses.
These are the primary signals for emotion in speech, which AST (audio event
classifier) cannot capture.

Model variants and hidden sizes:
    - openai/whisper-tiny:   384d (39M)
    - openai/whisper-base:   512d (74M)
    - openai/whisper-small:  768d (244M) ← default, matches d_model=768
    - openai/whisper-medium: 1024d (769M)
    - openai/whisper-large-v3: 1280d (1.5B)

Output: token sequence (B, target_tokens, d_out) where d_out defaults to the
model's native hidden size (self.d_out). Dimension standardization across
model variants (or across whisper/ast/wav2vec2/hubert during ablation) is
handled downstream by the fusion module's per-modality MLP (see
`fusion.audio_dim` in config.yaml / `modality_dim_kwargs`), NOT here — that
MLP is trained jointly with the rest of the model, whereas anything done
inside this frozen, `@torch.no_grad()`-wrapped encoder never receives
gradients. Only pass `d_out` explicitly if you understand this tradeoff.
"""

import logging
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoFeatureExtractor, WhisperModel

logger = logging.getLogger(__name__)


class WhisperAudioEncoder(nn.Module):
    """Whisper encoder wrapper for speech prosody features."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        target_tokens: int = 64,
        d_out: int | None = None,
        sample_rate: int = 16000,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize Whisper audio encoder.

        Args:
            model_name: HuggingFace Whisper model ID.
            target_tokens: Output sequence length after adaptive pooling.
            d_out: If set and different from the model's native hidden size,
                adds an (untrained, frozen-context) nn.Linear projection.
                Leave as None (default) to output the native hidden size and
                let `fusion.audio_dim` handle standardization instead.
            sample_rate: Input audio sample rate (must be 16kHz for Whisper).
            device: Torch device.
        """
        super().__init__()
        self.model_name = model_name
        self.target_tokens = target_tokens
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        logger.info("Loading Whisper encoder: %s", model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        del whisper.decoder
        del whisper

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder = self.encoder.to(self.device)

        d_model = self.encoder.config.d_model
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
        logger.info("Whisper encoder loaded and frozen (d_model=%d, d_out=%d)", d_model, self.d_out)

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
        input_features = inputs.input_features.to(self.device)

        hidden = self.encoder(input_features).last_hidden_state

        if self.proj is not None:
            hidden = self.proj(hidden)

        return self._adaptive_pool(hidden, self.target_tokens)

    @torch.no_grad()
    def encode_batch(self, audio_paths: list[Path]) -> Tensor:
        """Batch encode audio files.

        Args:
            audio_paths: List of wav file paths.

        Returns:
            Tensor of shape (N, target_tokens, d_out).
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
        pooled = F.adaptive_avg_pool1d(x_t, target_len)
        return pooled.transpose(1, 2)
