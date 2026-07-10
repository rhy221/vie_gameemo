"""Raw-audio augmentation, applied to the waveform BEFORE any encoder-specific
feature extraction (log-mel for Whisper/AST, raw normalization for
Wav2Vec2/HuBERT) — all 4 `audio_encoder` backends start from the same
`librosa.load(...)` waveform, so augmenting at this level works uniformly
across every backend without per-encoder hooks.

SpecAugment is normally applied to the exact spectrogram tensor a model
consumes. Since 2 of the 4 backends here (Wav2Vec2, HuBERT) take a raw
waveform with no spectrogram at all, `spec_augment_waveform` instead computes
a mel-spectrogram, masks it, and reconstructs a waveform (Griffin-Lim) — a
lossy round-trip, but it keeps one uniform "waveform in, waveform out"
augmentation pipeline across all backends instead of 4 divergent code paths.
"""

import random

import librosa
import numpy as np


def pitch_shift(wav: np.ndarray, sr: int, semitone_range: tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    """Randomly shift pitch by n semitones sampled from `semitone_range`."""
    n_steps = random.uniform(*semitone_range)
    if abs(n_steps) < 1e-6:
        return wav
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps)


def time_stretch(wav: np.ndarray, rate_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Randomly speed up/slow down, then pad/trim back to the original length
    so downstream duration-dependent logic (target_tokens, frame sampling)
    isn't affected by the augmentation.
    """
    rate = random.uniform(*rate_range)
    if abs(rate - 1.0) < 1e-6:
        return wav
    stretched = librosa.effects.time_stretch(wav, rate=rate)
    return _fix_length(stretched, len(wav))


def add_noise(wav: np.ndarray, std_range: tuple[float, float] = (0.0, 0.01)) -> np.ndarray:
    """Add Gaussian noise, std sampled uniformly from `std_range` (absolute
    amplitude — librosa waveforms are float32 in roughly [-1, 1])."""
    std = random.uniform(*std_range)
    if std <= 0:
        return wav
    return (wav + np.random.randn(*wav.shape).astype(wav.dtype) * std).astype(wav.dtype)


def spec_augment_waveform(
    wav: np.ndarray,
    sr: int,
    time_mask_p: float = 0.2,
    freq_mask_p: float = 0.2,
    max_time_mask_frac: float = 0.1,
    max_freq_mask_frac: float = 0.1,
    n_mels: int = 80,
) -> np.ndarray:
    """SpecAugment-style time/freq masking via a mel-spectrogram round-trip.

    Computes a mel-spectrogram, zeroes a random contiguous time span and/or
    frequency band, then reconstructs a waveform (Griffin-Lim) of the same
    length as the input. See module docstring for why this round-trip
    approach is used instead of masking a model-specific feature tensor.

    Args:
        wav: 1D float32 waveform.
        sr: Sample rate.
        time_mask_p / freq_mask_p: Probability of applying that mask.
        max_time_mask_frac / max_freq_mask_frac: Max fraction of that axis
            a single mask span can cover.
        n_mels: Mel bands for the intermediate spectrogram.

    Returns:
        Reconstructed waveform, same length as input.
    """
    if time_mask_p <= 0 and freq_mask_p <= 0:
        return wav

    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels)
    n_freq, n_time = mel.shape

    if freq_mask_p > 0 and random.random() < freq_mask_p and n_freq > 1:
        max_len = max(1, int(n_freq * max_freq_mask_frac))
        mask_len = random.randint(1, max_len)
        start = random.randint(0, n_freq - mask_len)
        mel[start:start + mask_len, :] = 0.0

    if time_mask_p > 0 and random.random() < time_mask_p and n_time > 1:
        max_len = max(1, int(n_time * max_time_mask_frac))
        mask_len = random.randint(1, max_len)
        start = random.randint(0, n_time - mask_len)
        mel[:, start:start + mask_len] = 0.0

    reconstructed = librosa.feature.inverse.mel_to_audio(mel, sr=sr, length=len(wav))
    return reconstructed.astype(wav.dtype)


def _fix_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim to exactly `target_len` samples."""
    if len(wav) == target_len:
        return wav
    if len(wav) > target_len:
        return wav[:target_len]
    return np.pad(wav, (0, target_len - len(wav)))


def augment_waveform(
    wav: np.ndarray,
    sr: int,
    pitch_shift_semitones: tuple[float, float] | None = None,
    time_stretch_rate: tuple[float, float] | None = None,
    noise_std: tuple[float, float] | None = None,
    spec_augment: dict | None = None,
) -> np.ndarray:
    """Apply the configured combination of waveform augmentations, in order:
    pitch shift -> time stretch -> SpecAugment round-trip -> additive noise
    (noise added last so it isn't partially smoothed away by the Griffin-Lim
    reconstruction step).

    Args:
        wav: 1D float32 waveform (as returned by `librosa.load`).
        sr: Sample rate.
        pitch_shift_semitones: Range for `pitch_shift`, or None to skip.
        time_stretch_rate: Range for `time_stretch`, or None to skip.
        noise_std: Range for `add_noise`, or None to skip.
        spec_augment: Kwargs for `spec_augment_waveform` (minus wav/sr), or None to skip.

    Returns:
        Augmented waveform, same length as input.
    """
    out = wav
    if pitch_shift_semitones is not None:
        out = pitch_shift(out, sr, pitch_shift_semitones)
    if time_stretch_rate is not None:
        out = time_stretch(out, time_stretch_rate)
    if spec_augment:
        out = spec_augment_waveform(out, sr, **spec_augment)
    if noise_std is not None:
        out = add_noise(out, noise_std)
    return out
