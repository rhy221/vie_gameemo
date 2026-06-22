"""Attention weight visualization for Conv-Attention fusion.

Plots modality attention weights (audio / face / context / text) over time
and across genres, enabling interpretability analysis. Key figures for the
report: which modality does the model rely on, and how does that vary by
game genre or emotion class.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_MODALITY_NAMES = ["audio", "face", "context", "text"]
_MODALITY_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
_EMOTION_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]


def plot_attention_timeline(
    attn_weights: Tensor | np.ndarray,
    clip_id: str = "",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> "Figure":
    """Plot modality attention weights over time for a single clip.

    Args:
        attn_weights: (T, 4) attention weights from ConvAttention4M.
        clip_id: Clip identifier for title.
        save_path: If provided, save figure to this path.
        figsize: Matplotlib figure size.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if isinstance(attn_weights, Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if attn_weights.ndim == 3:
        attn_weights = attn_weights[0]

    T, n_mods = attn_weights.shape
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(min(n_mods, len(_MODALITY_NAMES))):
        ax.plot(timesteps, attn_weights[:, i],
                label=_MODALITY_NAMES[i], color=_MODALITY_COLORS[i],
                linewidth=2, alpha=0.85)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Attention weight")
    ax.set_title(f"Modality attention over time — {clip_id}" if clip_id else
                 "Modality attention over time")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Attention timeline saved → %s", save_path)

    return fig


def plot_attention_stacked(
    attn_weights: Tensor | np.ndarray,
    clip_id: str = "",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> "Figure":
    """Stacked area plot of modality attention over time.

    Args:
        attn_weights: (T, 4) attention weights.
        clip_id: Clip identifier for title.
        save_path: Optional save path.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if isinstance(attn_weights, Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if attn_weights.ndim == 3:
        attn_weights = attn_weights[0]

    T, n_mods = attn_weights.shape
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)
    labels = _MODALITY_NAMES[:n_mods]
    colors = _MODALITY_COLORS[:n_mods]
    ax.stackplot(timesteps, *[attn_weights[:, i] for i in range(n_mods)],
                 labels=labels, colors=colors, alpha=0.8)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Attention weight")
    ax.set_title(f"Modality attention distribution — {clip_id}" if clip_id else
                 "Modality attention distribution")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_mean_attention_by_genre(
    predictions: list[dict],
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> "Figure":
    """Bar chart: mean modality attention per genre.

    Args:
        predictions: List of dicts with 'genre' and 'attn_weights' (T, 4) keys.
        save_path: Optional save path.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    genre_weights: dict[str, list[np.ndarray]] = defaultdict(list)
    for pred in predictions:
        genre = pred.get("genre", "unknown")
        w = pred.get("attn_weights")
        if w is None:
            continue
        if isinstance(w, Tensor):
            w = w.detach().cpu().numpy()
        if w.ndim == 3:
            w = w[0]
        genre_weights[genre].append(w.mean(axis=0))

    if not genre_weights:
        logger.warning("No attention weights found in predictions")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    genres = sorted(genre_weights.keys())
    n_genres = len(genres)
    n_mods = len(_MODALITY_NAMES)
    means = np.zeros((n_genres, n_mods))
    for i, genre in enumerate(genres):
        stacked = np.stack(genre_weights[genre])
        means[i] = stacked.mean(axis=0)

    x = np.arange(n_genres)
    width = 0.8 / n_mods

    fig, ax = plt.subplots(figsize=figsize)
    for m in range(n_mods):
        offset = (m - n_mods / 2 + 0.5) * width
        ax.bar(x + offset, means[:, m], width,
               label=_MODALITY_NAMES[m], color=_MODALITY_COLORS[m], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(genres)
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Modality attention by game genre")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Genre attention plot saved → %s", save_path)

    return fig


def plot_mean_attention_by_emotion(
    predictions: list[dict],
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> "Figure":
    """Bar chart: mean modality attention per emotion class.

    Args:
        predictions: List of dicts with 'y_true' (int or str) and 'attn_weights'.
        save_path: Optional save path.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    label2idx = {l: i for i, l in enumerate(_EMOTION_LABELS)}
    emotion_weights: dict[str, list[np.ndarray]] = defaultdict(list)

    for pred in predictions:
        y = pred.get("y_true", pred.get("label", None))
        if y is None:
            continue
        if isinstance(y, int) and y < len(_EMOTION_LABELS):
            label = _EMOTION_LABELS[y]
        else:
            label = str(y)

        w = pred.get("attn_weights")
        if w is None:
            continue
        if isinstance(w, Tensor):
            w = w.detach().cpu().numpy()
        if w.ndim == 3:
            w = w[0]
        emotion_weights[label].append(w.mean(axis=0))

    if not emotion_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    emotions = [e for e in _EMOTION_LABELS if e in emotion_weights]
    n_emo = len(emotions)
    n_mods = len(_MODALITY_NAMES)
    means = np.zeros((n_emo, n_mods))
    for i, emo in enumerate(emotions):
        stacked = np.stack(emotion_weights[emo])
        means[i] = stacked.mean(axis=0)

    x = np.arange(n_emo)
    width = 0.8 / n_mods

    fig, ax = plt.subplots(figsize=figsize)
    for m in range(n_mods):
        offset = (m - n_mods / 2 + 0.5) * width
        ax.bar(x + offset, means[:, m], width,
               label=_MODALITY_NAMES[m], color=_MODALITY_COLORS[m], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=30, ha="right")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Modality attention by emotion class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Emotion attention plot saved → %s", save_path)

    return fig


def collect_attention_weights(
    fusion: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: "DataLoader",
    device: torch.device,
) -> list[dict]:
    """Run inference and collect attention weights from fusion module.

    Args:
        fusion: Fusion module (must have return_attention=True).
        classifier: Classifier module.
        loader: DataLoader yielding batches.
        device: Torch device.

    Returns:
        List of dicts with keys: clip_id, genre, y_true, y_pred, attn_weights.
    """
    fusion.eval()
    classifier.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            context = batch["context"].to(device)
            text = batch["text"].to(device)
            labels = batch["label"].to(device)
            has_face = batch.get("has_face")
            if has_face is not None:
                has_face = has_face.to(device)

            out = fusion(audio, face, context, text, has_face=has_face)
            if isinstance(out, tuple) and len(out) == 2:
                fused, attn_weights = out
            else:
                fused = out
                attn_weights = None

            logits = classifier(fused)
            preds = logits.argmax(dim=-1)

            clip_ids = batch.get("clip_id", [""] * len(labels))
            genres = batch.get("genre", ["unknown"] * len(labels))

            for i in range(len(labels)):
                entry = {
                    "clip_id": clip_ids[i] if isinstance(clip_ids, (list, tuple)) else "",
                    "genre": genres[i] if isinstance(genres, (list, tuple)) else "unknown",
                    "y_true": int(labels[i].item()),
                    "y_pred": int(preds[i].item()),
                }
                if attn_weights is not None:
                    entry["attn_weights"] = attn_weights[i].cpu().numpy()
                results.append(entry)

    return results


def save_attention_report(
    predictions: list[dict],
    output_dir: Path,
) -> None:
    """Generate and save all attention visualization plots.

    Args:
        predictions: Output of collect_attention_weights.
        output_dir: Directory to save plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_attention_by_genre(predictions, save_path=output_dir / "attn_by_genre.png")
    plot_mean_attention_by_emotion(predictions, save_path=output_dir / "attn_by_emotion.png")

    for i, pred in enumerate(predictions[:5]):
        w = pred.get("attn_weights")
        if w is not None:
            clip_id = pred.get("clip_id", f"sample_{i}")
            plot_attention_timeline(w, clip_id=clip_id,
                                   save_path=output_dir / f"timeline_{clip_id}.png")
            plot_attention_stacked(w, clip_id=clip_id,
                                  save_path=output_dir / f"stacked_{clip_id}.png")

    plt.close("all")
    logger.info("Attention report saved → %s", output_dir)
