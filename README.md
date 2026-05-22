# Vie-GameEmo

Multimodal emotion recognition for Vietnamese game livestreams/reviews.

Combines **audio** (spectrogram → AST), **visual dual-path** (streamer face crop + gameplay context), and **text** (Whisper → XLM-RoBERTa) modalities via Conv-Attention fusion, with LLM-based reasoning explanations in Vietnamese.

## Why dual-path visual?

Most multimodal emotion recognition models (e.g., Emotion-LLaMAv2) work on talking-head videos where the face occupies 60-85% of the frame. **Livestream gaming is different**: the streamer's webcam is only 10-15% of the frame, with the rest being gameplay. Naively applying full-frame ViT-FER suffers from:

1. **Diluted signal**: face is a tiny fraction of pixels
2. **Confusion with in-game NPC faces** (cutscenes in Horror/RPG)
3. **Distribution shift** from AffectNet pretrain

Our **dual-path** design solves this:
- **Path 1 (Face)**: detect webcam region with stable clustering (DBSCAN over MediaPipe detections), crop streamer face, encode with ViT-FER
- **Path 2 (Context)**: encode full frame with generic ViT-ImageNet to retain gameplay context (boss fights, kill feeds, etc.)

Both paths feed into 4-modality Conv-Attention fusion.

## Quick Start

### Install (cross-platform)

Requires Python 3.10+ and (recommended) a CUDA-capable GPU.

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA torch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

External dependency: **OpenFace 2.x** for AU extraction.
Install from https://github.com/TadasBaltrusaitis/OpenFace, set `annotation.openface.binary_path` in `config.yaml`.

### Example workflows

#### 1. Stage 0 — Data preparation (runs INDEPENDENTLY)

```bash
# Crawl videos from URL list
python scripts/stage0_crawl.py --config config.yaml

# Extract audio + frames + detect webcam region
python scripts/stage0_preprocess.py --config config.yaml

# Multi-agent annotation (heavy: A100 recommended, or T4 with smaller models)
python scripts/stage0_annotate.py --config config.yaml --labels-csv data/annotations/labels.csv
```

#### 2. Feature extraction (cache for fast training)

```bash
python scripts/extract_features.py --config config.yaml
```

#### 3. Train (curriculum)

```bash
# Stage 1 — Perception (fusion + classifier)
python scripts/train.py --config config.yaml --stage perception

# Stage 2 — Cognition (joint cls + reasoning, requires perception checkpoint)
python scripts/train.py --config config.yaml --stage cognition \
    --resume-from outputs/checkpoints/perception_best.pt
```

#### 4. RLVR for LLM-4 (optional, R1-Omni-inspired)

```bash
# Cold start (brief SFT)
python scripts/train_rlvr.py --config config.yaml --phase cold-start

# RLVR (long, A100 recommended)
python scripts/train_rlvr.py --config config.yaml --phase rlvr \
    --resume-from outputs/checkpoints/llm4_coldstart.pt
```

#### 5. Evaluate

```bash
# Standard eval
python scripts/eval.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt --split test_id

# Strategy A vs B vs C ablation (full_frame / face_only / dual_path)
python scripts/eval.py --config config.yaml --ablation strategy

# Per-genre breakdown
python scripts/eval.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt --per-genre
```

#### 6. Inference + Demo

```bash
# Single clip
python scripts/infer.py --config config.yaml \
    --checkpoint outputs/checkpoints/best.pt --input clip.mp4

# Gradio demo
python scripts/demo.py --config config.yaml \
    --checkpoint outputs/checkpoints/best.pt --mode batch
```

## Project Structure

```
vie-gameemo/
├── config.yaml                   # Main config (all hyperparameters)
├── configs/experiments/          # Experiment overrides (ablation runs)
├── src/vie_gameemo/
│   ├── data/                     # Schemas, crawler, dataset, multi-agent annotator
│   ├── preprocess/               # Demux, webcam detection, face crop
│   ├── encoders/                 # AST audio, ViT-FER face, ViT-ImageNet context, XLM-R text
│   ├── fusion/                   # Conv-Attention 4-modality + baselines
│   ├── classifiers/              # MLP emotion classifier
│   ├── llm/                      # LLM-1/2/3/4 reasoner setups
│   ├── training/                 # Perception + Cognition trainers
│   ├── evaluation/               # Metrics, strategy ablation, per-genre
│   ├── inference/                # Batch + realtime
│   └── utils/                    # Config, logging, seed, I/O
├── scripts/                      # CLI entry points (one per stage/task)
├── data/                         # Raw, processed, annotations, features (.gitignore'd)
├── outputs/                      # Checkpoints, logs, results
└── notebooks/                    # EDA, exploration
```

## Configuration

All settings live in `config.yaml`. Override per-experiment via `configs/experiments/*.yaml` (deep merge), or per-run via CLI args (highest precedence).

Example experiment configs:
- `strategy_a_full_frame.yaml` — paper baseline (no webcam detection)
- `strategy_b_face_only.yaml` — face crop only (no context path)
- `strategy_c_dual_path.yaml` — recommended (default)
- `ablation_fusion.yaml` — compare fusion modules
- `ablation_llm.yaml` — compare 4 LLM setups

## Compute Profiles

| Profile | Annotation consolidator | LLM-4 RLVR base | Hardware |
|---|---|---|---|
| **A100** (recommended) | Qwen2.5-72B q4 | Qwen2.5-7B | 1× A100 80GB |
| **Colab/Kaggle** | Qwen2.5-7B q4 | Qwen2.5-1.5B or 0.5B | T4/L4 16GB |

Set `compute.profile` in `config.yaml`.

## Stage Independence

**Stage 0 runs independently from training.** This is intentional:

- Stage 0 (annotation) is expensive (days) and needs different compute (large LLMs).
- Stage 1-5 (training) is iterative (many runs for ablations).
- A user can run only Stage 0 on a rented A100, transfer artifacts (JSON + audio/frames), then run training on Colab.

Each stage has its own CLI; no Python imports cross the boundary.

## Citation

If you use this work, please cite (TODO: paper bibtex).

## License

MIT (or as specified).
