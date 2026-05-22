"""Audio encoder using AST (Audio Spectrogram Transformer).

AST is pretrained on AudioSet (audio event classification). It treats the
log-mel spectrogram as an image and applies a ViT-style transformer.
For our use case, AST captures non-speech audio events (laughs, shouts,
sighs) that complement Whisper's speech transcription.

Output: token sequence (B, target_tokens, 768), adaptively pooled.
"""

import logging
from pathlib import Path

import librosa
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
        sample_rate: int = 16000,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize AST encoder.

        Args:
            model_name: HuggingFace model ID.
            target_tokens: Output sequence length after adaptive pooling.
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
        logger.info("AST encoder loaded and frozen")

    @torch.no_grad()
    def encode(self, audio_path: Path) -> Tensor:
        """Encode a single audio file.

        Args:
            audio_path: Path to wav file (sample_rate Hz, mono).

        Returns:
            Tensor of shape (1, target_tokens, 768).

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, N, 768)
        return self._adaptive_pool(hidden, self.target_tokens)

    @torch.no_grad()
    def encode_batch(self, audio_paths: list[Path]) -> Tensor:
        """Batch encode audio files.

        Args:
            audio_paths: List of wav file paths.

        Returns:
            Tensor of shape (N, target_tokens, 768).
        """
        tensors = [self.encode(p) for p in audio_paths]  # each (1, T, 768)
        return torch.cat(tensors, dim=0)  # (N, T, 768)

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
