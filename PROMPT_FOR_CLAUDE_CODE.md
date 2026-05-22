# Prompt for Claude Code: Implement Vie-GameEmo Pipeline

## Context

You are implementing **Vie-GameEmo** — a multimodal emotion recognition system for Vietnamese game livestreams/reviews. The system combines:
- **Audio** → log-mel spectrogram → AST encoder
- **Visual dual-path**: streamer face (crop from webcam region) + gameplay context (full-frame)
- **Text** → Whisper ASR → XLM-RoBERTa encoder

These 4 modalities are fused via a **Conv-Attention** module, classified for emotion label, and explained by an LLM (4 setups to compare).

**Source of truth:** `pipeline_implementation_detailed.md` (provided alongside this prompt). When this prompt and the spec differ:
- This prompt wins for **scope, order, conventions**
- The spec wins for **technical details, architecture, hyperparameters**

**Your scope:** Implement **all stages 0–5** with the dual-path visual approach and the 4-LLM-setup comparison.

---

## Setup Provided

You have:
1. **`PROMPT_FOR_CLAUDE_CODE.md`** — this file
2. **`pipeline_implementation_detailed.md`** — full design spec (~2100 lines)
3. **`config.yaml`** — main configuration file (already filled)
4. **Project skeleton** — folder structure with stub `.py` files containing docstrings and TODO markers

**Do not change the top-level folder structure** unless absolutely necessary. Implement code inside the stubs.

---

## Coding Standards

### Style
- **Python 3.10+** required (use modern type hints: `list[int]`, `X | None` instead of `Optional[X]`)
- **Type hints** on all public functions, classes, and methods
- **Google-style docstrings** for everything public, including a one-line summary
- **No `print()` for logs** — use the `logger` from `vie_gameemo.utils.logging`
- **No global mutable state**
- **Line length:** 100 chars
- **Imports:** stdlib → third-party → local, separated by blank lines

### Error Handling
- Validate inputs at module boundaries; raise `ValueError` / `FileNotFoundError` with informative messages
- Never silently swallow exceptions; log at minimum, re-raise where appropriate
- For long-running ops (annotation, training), checkpoint progress and resume on restart

### Determinism
- Always call `set_seed()` from `vie_gameemo.utils.seed` at the start of every entry point
- Avoid non-deterministic ops unless flagged

### CLI Convention
- Each entry point script under `scripts/` uses **argparse** (no Hydra, no Click)
- Common args pattern:
  ```bash
  python scripts/stage0_crawl.py --config config.yaml --output data/raw_videos
  python scripts/train.py --config config.yaml --stage perception
  python scripts/eval.py --config config.yaml --checkpoint outputs/checkpoints/best.pt
  ```
- All scripts must support `--config PATH` and `--help`
- Output paths come from config; CLI args override config

### Cross-Platform
- Use `pathlib.Path` everywhere, never raw string paths or `os.path.join`
- File operations use UTF-8 explicitly: `open(p, encoding="utf-8")`
- Subprocess calls (ffmpeg, etc.) use a list of args, not shell=True

---

## Stage Independence (CRITICAL)

**Stage 0 must run independently of the model pipeline.**

The reasoning:
- Stage 0 (data prep + multi-agent annotation) is expensive (days), needs different compute (large LLMs for annotation)
- Stage 1-5 (preprocessing + model + training) is iterative (many runs for ablation)
- These should not be coupled

**Concretely:**
- Stage 0 scripts (`scripts/stage0_*.py`) write outputs to disk as **standalone artifacts**: clip files, JSON annotations, transcripts
- Stage 1-5 scripts **read those artifacts from disk**, never call Stage 0 code internally
- A user must be able to:
  - Run only Stage 0 on a beefy machine (e.g., rented A100), then transfer artifacts to Colab
  - Run Stage 1-5 on Colab without ever invoking Stage 0
  - Re-run an individual stage (e.g., re-train fusion module) without touching others

**Each stage has its own CLI entry point** under `scripts/`:
- `scripts/stage0_crawl.py` — download videos from URL list
- `scripts/stage0_preprocess.py` — extract audio/frames (Stage 1 in spec, grouped with 0 for data prep)
- `scripts/stage0_annotate.py` — multi-agent annotation pipeline
- `scripts/stage0_verify.py` — human verification helper
- `scripts/extract_features.py` — run encoders (Stage 2a/b/c) and cache features (used by Stage 3-4 training)
- `scripts/train.py` — Stage 3-4 training (perception + cognition curriculum)
- `scripts/train_rlvr.py` — Stage 5 RLVR (separate due to compute profile)
- `scripts/eval.py` — evaluation
- `scripts/infer.py` — single-clip or batch inference
- `scripts/demo.py` — Gradio demo

---

## Implementation Order

Follow this order strictly; do not jump ahead. After each phase, the project should be runnable end-to-end up to that point.

### Phase 1 — Foundation (do first)
1. `src/vie_gameemo/utils/{logging.py, seed.py, config.py, io.py}` — config loader, logger, seed control, file helpers
2. `src/vie_gameemo/data/schemas.py` — Pydantic models for `Clip`, `Annotation`, `MultimodalFeatures`, `EmotionLabel`
3. Verify: `python -c "from vie_gameemo.utils.config import load_config; print(load_config('config.yaml').data.dataset_size)"` runs without error

### Phase 2 — Stage 0: Data Pipeline
4. `src/vie_gameemo/data/crawler.py` — yt-dlp wrapper
5. `src/vie_gameemo/preprocess/demux.py` — ffmpeg audio/frame extraction
6. `src/vie_gameemo/preprocess/webcam_detector.py` — MediaPipe-based webcam region detection (Section 5.3 of spec)
7. `src/vie_gameemo/preprocess/face_crop.py` — face crop from webcam region
8. `src/vie_gameemo/data/annotator/` — multi-agent annotation modules:
   - `openface_au.py` — AU extraction
   - `peak_frame.py` — peak frame detection
   - `qwen_vl_agent.py` — visual objective description
   - `qwen_audio_agent.py` — audio tone description
   - `whisper_asr.py` — transcript
   - `consolidator.py` — Qwen2.5-32B/72B consolidator
   - `pipeline.py` — orchestrator that calls all agents serially (with model load/unload for memory)
9. `scripts/stage0_crawl.py`, `scripts/stage0_preprocess.py`, `scripts/stage0_annotate.py`
10. Verify: each script runs with `--help` and produces correct artifacts on a small input

### Phase 3 — Stage 2 Encoders
11. `src/vie_gameemo/encoders/audio_ast.py` — AST encoder with adaptive pooling to 64 tokens
12. `src/vie_gameemo/encoders/face_vit.py` — Face encoder (dual-view: global + temporal)
13. `src/vie_gameemo/encoders/context_vit.py` — Context encoder (full-frame ViT-ImageNet)
14. `src/vie_gameemo/encoders/text_xlmr.py` — XLM-RoBERTa encoder
15. `src/vie_gameemo/data/dataset.py` — PyTorch `Dataset` reading from annotations + raw artifacts
16. `src/vie_gameemo/data/feature_cache.py` — precompute and cache features (frozen encoders → train fast)
17. `scripts/extract_features.py` — CLI to run encoders and cache
18. Verify: produces `.pt` files under `data/features/`

### Phase 4 — Stage 3 Fusion
19. `src/vie_gameemo/fusion/conv_attention.py` — Conv-Attention 4-modality module (Section 7.3 of spec)
20. `src/vie_gameemo/fusion/baselines.py` — Late, Early, MULT, Q-Former (for ablation)
21. `src/vie_gameemo/fusion/__init__.py` — factory `get_fusion(name)`

### Phase 5 — Stage 4 Classifier + Stage 1 Training
22. `src/vie_gameemo/classifiers/mlp.py` — emotion classifier MLP
23. `src/vie_gameemo/training/losses.py` — Focal Loss, weighted CE
24. `src/vie_gameemo/training/perception.py` — Stage 1 (Perception) trainer
25. `src/vie_gameemo/training/cognition.py` — Stage 2 (Cognition) trainer (joint cls + reasoning LM)
26. `scripts/train.py` — CLI dispatching to perception/cognition trainer
27. Verify: `python scripts/train.py --stage perception --config config.yaml` trains on cached features

### Phase 6 — Stage 5 LLM (4 setups)
28. `src/vie_gameemo/llm/base.py` — abstract base class for LLM setups
29. `src/vie_gameemo/llm/llm1_explainer.py` — post-hoc explainer (no training, prompt only)
30. `src/vie_gameemo/llm/llm2_coreasoner.py` — modality-to-text + LLM reasoner
31. `src/vie_gameemo/llm/llm3_vlm.py` — VLM end-to-end with LoRA fine-tune
32. `src/vie_gameemo/llm/llm4_rlvr.py` — RLVR with GRPO (cold start + reward func)
33. `scripts/train_rlvr.py` — RLVR-specific trainer (uses TRL library)

### Phase 7 — Evaluation + Inference + Demo
34. `src/vie_gameemo/evaluation/metrics.py` — accuracy, F1, UAR, per-class
35. `src/vie_gameemo/evaluation/strategy_ablation.py` — Strategy A/B/C comparison logic
36. `src/vie_gameemo/evaluation/per_genre.py` — per-genre breakdown
37. `src/vie_gameemo/inference/batch.py` — batch inference
38. `src/vie_gameemo/inference/realtime.py` — real-time inference with sliding window
39. `scripts/eval.py`, `scripts/infer.py`, `scripts/demo.py`

---

## Key Technical Decisions (Already Made)

These come from the spec; do not change unless you have a strong reason and flag it:

| Decision | Choice | Spec section |
|---|---|---|
| Audio encoder | AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) | 4.2 |
| Audio tokens | 64 (adaptive pool) | 4.2 |
| Visual approach | **Dual-path** (face crop + context full-frame), NOT full-frame only | 5 (Section 5.1 explains why) |
| Webcam detection | MediaPipe + DBSCAN clustering for stability | 5.3 |
| Face encoder | ViT-FER (`trpakov/vit-face-expression`) | 5.3 |
| Context encoder | ViT-ImageNet (`google/vit-base-patch16-224`) | 5.4 |
| Text encoder | XLM-RoBERTa-base | 6.3 |
| ASR | Whisper-large-v3 via faster-whisper | 6.1 |
| Fusion | **Conv-Attention 4-modality** (audio + face + context + text) | 7 |
| Training | **Curriculum: Perception → Cognition** | 10 |
| LLM base | Qwen2.5-7B-Instruct (4-bit), fallback Qwen2.5-1.5B/0.5B | 9 |
| Annotation | Multi-agent with serial model load/unload | 11.1 |
| Reproducibility | seed everywhere, log seed to wandb | 11.7 |

---

## Code Examples (Style Reference)

### Good
```python
"""Audio encoder using AST (Audio Spectrogram Transformer).

Loads log-mel spectrogram, encodes via AST, and pools to a fixed token length.
"""

import logging
from pathlib import Path

import librosa
import torch
from torch import Tensor
from transformers import ASTFeatureExtractor, ASTModel

from vie_gameemo.utils.config import AudioConfig

logger = logging.getLogger(__name__)


class ASTEncoder:
    """Wraps AST for emotion-aware audio encoding.

    Args:
        cfg: Audio configuration (sample rate, target token length, model name).
        device: Torch device.

    Attributes:
        model: Frozen AST backbone.
        target_tokens: Number of output tokens after adaptive pooling.
    """

    def __init__(self, cfg: AudioConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.target_tokens = cfg.target_tokens

        logger.info("Loading AST model: %s", cfg.model_name)
        self.extractor = ASTFeatureExtractor.from_pretrained(cfg.model_name)
        self.model = ASTModel.from_pretrained(cfg.model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, audio_path: Path) -> Tensor:
        """Encode an audio file to a fixed-length token sequence.

        Args:
            audio_path: Path to a wav file (16kHz, mono).

        Returns:
            Tensor of shape (1, target_tokens, 768).

        Raises:
            FileNotFoundError: If audio_path does not exist.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, _ = librosa.load(str(audio_path), sr=self.cfg.sample_rate, mono=True)
        inputs = self.extractor(audio, sampling_rate=self.cfg.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, N, 768)
        return self._adaptive_pool(hidden, self.target_tokens)

    @staticmethod
    def _adaptive_pool(x: Tensor, target_len: int) -> Tensor:
        """Adaptive average pool along the sequence dim to target_len."""
        x_t = x.transpose(1, 2)
        pooled = torch.nn.functional.adaptive_avg_pool1d(x_t, target_len)
        return pooled.transpose(1, 2)
```

### Bad (avoid)
```python
# No types, print as logging, raw path strings, no docstring
def encode(audio_path):
    print("encoding", audio_path)
    audio, _ = librosa.load(audio_path)
    ...
```

---

## Memory Management (Annotation Pipeline)

The multi-agent annotation pipeline loads several large LLMs. To fit on consumer GPUs (T4/L4), use **serial load/unload**:

```python
# In data/annotator/pipeline.py
def annotate_batch(clip_paths: list[Path], cfg: AnnotationConfig) -> list[Annotation]:
    """Batch annotate clips. Phases run serially to manage VRAM."""

    # Phase 1: lightweight local ops (CPU/small GPU)
    quick = _phase_local(clip_paths)

    # Phase 2: Qwen2.5-VL for visual descriptions
    vl_agent = QwenVLAgent(cfg.vl)
    cvods = vl_agent.batch_describe([q.peak_frame for q in quick])
    del vl_agent
    torch.cuda.empty_cache()

    # Phase 3: Qwen2-Audio for prosody
    audio_agent = QwenAudioAgent(cfg.audio)
    catds = audio_agent.batch_describe(clip_paths)
    del audio_agent
    torch.cuda.empty_cache()

    # Phase 4: Consolidator
    consolidator = Consolidator(cfg.consolidator)
    reasonings = consolidator.batch_consolidate(...)
    del consolidator
    torch.cuda.empty_cache()

    return [_compose(...) for ...]
```

Configure batch size from `config.yaml` (`annotation.batch_size`).

---

## Acceptance Criteria

Your implementation passes if:

1. **Project installs cleanly:** `pip install -r requirements.txt` succeeds on a fresh venv on both Linux and Windows
2. **All scripts respond to `--help`** with informative output
3. **Phase 1-2 verification:** running `scripts/stage0_*.py` on 5 sample clips produces expected artifacts (audio.wav, frames, annotations.json) under `data/processed/` and `data/annotations/`
4. **Phase 3 verification:** `scripts/extract_features.py` produces feature `.pt` files; loading them in Python yields tensors of expected shapes
5. **Phase 5 verification:** `scripts/train.py --stage perception --config config.yaml --epochs 1` completes one epoch on cached features without errors (even if accuracy is random — we are checking the pipeline, not convergence)
6. **Phase 6 verification:** `scripts/train.py --stage cognition` runs; each LLM setup (LLM-1, LLM-2, LLM-3, LLM-4) can be invoked via the CLI
7. **All public functions have type hints + Google-style docstrings**
8. **No `print()` statements** in `src/` (use logger)
9. **No hardcoded paths** in `src/` (use config)

---

## What NOT to Do

- Do NOT add a frontend framework (React, Vue) unless explicitly asked
- Do NOT add Docker/CI/CD unless explicitly asked
- Do NOT change the public CLI signatures listed above
- Do NOT couple Stage 0 with Stage 1-5 — they must be invocable independently
- Do NOT use full-frame visual encoding for emotion — use the dual-path approach (see Section 5 of spec for rationale)
- Do NOT skip docstrings or type hints
- Do NOT use synchronous downloads of huge models in `__init__` of test fixtures
- Do NOT swallow exceptions silently

---

## What to Do When Stuck

If something in the spec is ambiguous:
1. Pick the more conservative interpretation
2. Add a `# NOTE: assumption — ...` comment explaining your choice
3. List your assumptions at the end of your implementation summary

If a dependency is missing in `requirements.txt`, add it with a justifying comment.

If you need to change a hyperparameter that affects API, ask before doing so.

---

## Deliverables Summary

After all phases:

```
vie-gameemo/
├── README.md                     # Quick start, usage examples
├── requirements.txt              # All deps with pinned versions
├── config.yaml                   # Provided, may need additions
├── configs/experiments/*.yaml    # Stubs filled in (ablation overrides)
├── src/vie_gameemo/              # All modules implemented
├── scripts/                      # All CLI entry points working
├── notebooks/                    # 1-2 example notebooks (EDA, results)
└── data/, outputs/               # Empty dirs with .gitkeep
```

Final step: update `README.md` with:
- 1-paragraph project description
- Install instructions (pip + venv, both platforms)
- 3 example commands: Stage 0 annotate, train perception, eval
- Brief explanation of dual-path approach (why face crop + context, not full-frame)

---

Now begin **Phase 1**. After each phase, briefly summarize what you did and any assumptions made.
