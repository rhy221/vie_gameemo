# Hướng dẫn chạy Vie-GameEmo

Tài liệu này mô tả từng bước để cài đặt và chạy toàn bộ pipeline nhận diện cảm xúc cho livestream game tiếng Việt.

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Cấu hình](#3-cấu-hình)
4. [Stage 0 — Thu thập và gán nhãn dữ liệu](#4-stage-0--thu-thập-và-gán-nhãn-dữ-liệu)
5. [Stage 1–2 — Trích xuất đặc trưng](#5-stage-12--trích-xuất-đặc-trưng)
6. [Stage 3–4 — Huấn luyện mô hình (Perception)](#6-stage-34--huấn-luyện-mô-hình-perception)
7. [Stage 5 — Huấn luyện LLM Reasoner (Cognition + RLVR)](#7-stage-5--huấn-luyện-llm-reasoner-cognition--rlvr)
8. [Đánh giá](#8-đánh-giá)
9. [Inference và Demo](#9-inference-và-demo)
10. [Cấu hình phần cứng (Profiles)](#10-cấu-hình-phần-cứng-profiles)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Yêu cầu hệ thống

| Yêu cầu | Tối thiểu | Khuyến nghị |
|---|---|---|
| Python | 3.10+ | 3.11 |
| VRAM GPU | 8 GB (fp16) | 16 GB (T4/L4) hoặc 80 GB (A100) |
| RAM | 16 GB | 32 GB |
| Ổ đĩa | 100 GB | 500 GB (dữ liệu video raw) |
| OS | Linux / Windows / macOS | Linux (Ubuntu 20.04+) |

**Phụ thuộc ngoài (cần cài riêng):**
- [OpenFace 2.x](https://github.com/TadasBaltrusaitis/OpenFace) — trích xuất Action Units từ khuôn mặt
- [FFmpeg](https://ffmpeg.org/) — demux audio + frames từ video
- CUDA Toolkit 12.1+ (nếu dùng GPU NVIDIA)

---

## 2. Cài đặt môi trường

### 2.1 Tạo môi trường Python

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2.2 Cài PyTorch với CUDA

```bash
# CUDA 12.1 (khuyến nghị)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (chậm, không khuyến nghị cho training)
pip install torch torchvision torchaudio
```

### 2.3 Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `vllm` (trong requirements) yêu cầu Linux + CUDA. Nếu không cần RLVR, bỏ dòng `vllm` trong `requirements.txt` trước khi cài.

### 2.4 Cài OpenFace

```bash
# Ubuntu
sudo apt-get install -y libopencv-dev cmake
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
./download_models.sh
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(nproc)
```

Sau đó cập nhật đường dẫn trong `config.yaml`:
```yaml
annotation:
  openface:
    binary_path: "OpenFace/build/bin/FeatureExtraction"
```

### 2.5 Kiểm tra cài đặt

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import sys; sys.path.insert(0,'src'); from vie_gameemo.utils.config import load_config; cfg = load_config('config.yaml'); print('Config OK:', cfg.project.name)"
```

---

## 3. Cấu hình

Tất cả cài đặt nằm trong `config.yaml`. Không cần sửa code — chỉ sửa config.

### 3.1 Các tham số quan trọng cần kiểm tra

```yaml
# config.yaml

# Seed tái lặp kết quả
seed: 42

# Đường dẫn dữ liệu (tương đối so với thư mục gốc)
paths:
  raw_videos: "data/raw_videos"
  annotations: "data/annotations"
  features: "data/features"
  checkpoints: "outputs/checkpoints"

# Cấu hình phần cứng
compute:
  profile: "colab"   # "colab" (T4/L4) hoặc "a100"

# LLM đang dùng
llm:
  active_setup: "llm1"   # "llm1" | "llm2" | "llm3" | "llm4"
  base_model:
    name: "Qwen/Qwen2.5-7B-Instruct"
    quantization: "4bit"

# OpenFace binary
annotation:
  openface:
    binary_path: "OpenFace/build/bin/FeatureExtraction"
```

### 3.2 Override cho từng thí nghiệm

```bash
# Dùng file experiment override (deep merge)
python scripts/train.py --config config.yaml --experiment configs/experiments/strategy_c_dual_path.yaml

# Override trực tiếp từ CLI (ưu tiên cao nhất)
python scripts/train.py --config config.yaml training.perception.epochs=5 fusion.type=late
```

---

## 4. Stage 0 — Thu thập và gán nhãn dữ liệu

> **Stage 0 chạy độc lập với training.** Có thể chạy trên máy khác, sau đó copy artifacts (JSON + audio/frames) sang máy training.

### 4.1 Chuẩn bị danh sách URL

Tạo file `data/source_urls.txt`, mỗi dòng một URL YouTube:
```
https://www.youtube.com/watch?v=XXXXXXX
https://www.youtube.com/watch?v=YYYYYYY
```

Hoặc dùng TSV với thông tin streamer/genre:
```
url	streamer	genre
https://www.youtube.com/watch?v=XXXXXXX	streamer1	moba
https://www.youtube.com/watch?v=YYYYYYY	streamer2	fps
```

### 4.2 Tải video (Stage 0a)

```bash
python scripts/stage0_crawl.py --config config.yaml

# Tùy chọn
python scripts/stage0_crawl.py --config config.yaml \
    --source-list data/source_urls.txt \
    --output-dir data/raw_videos \
    --max-videos 100
```

Video được lưu tại `data/raw_videos/<streamer>/<video_id>.mp4`.

### 4.3 Tiền xử lý (Stage 0b)

```bash
python scripts/stage0_preprocess.py --config config.yaml

# Chỉ xử lý một số video
python scripts/stage0_preprocess.py --config config.yaml --videos-dir data/raw_videos/streamer1
```

Kết quả:
- Audio (16kHz mono WAV): `data/processed/audios/`
- Frames (4fps JPG): `data/processed/frames/`
- Webcam region JSON: `data/processed/webcam_bboxes/`
- Face crops: `data/processed/faces/`

### 4.4 Gán nhãn cảm xúc

Tạo file CSV với nhãn thủ công (từ người gán nhãn):
```csv
clip_id,emotion_label
streamer1_clip_001,hype
streamer1_clip_002,tilted
streamer2_clip_001,neutral
```

Nhãn hợp lệ (schema `gaming_8`): `neutral`, `hype`, `amused`, `tilted`, `sad`, `shocked`, `fear`, `disgusted`.
Xem `docs/annotation_guideline.md` để biết định nghĩa, AU đặc trưng, và quyết định khi mơ hồ.

### 4.5 Chạy pipeline gán nhãn đa tác tử (Stage 0c)

```bash
# Annotation đầy đủ (nặng, cần GPU ≥ 16GB VRAM)
python scripts/stage0_annotate.py --config config.yaml \
    --labels-csv data/annotations/labels.csv

# Chỉ một số clip để kiểm tra
python scripts/stage0_annotate.py --config config.yaml \
    --labels-csv data/annotations/labels.csv \
    --limit 10

# Resume nếu bị gián đoạn (tự động skip clip đã xong)
python scripts/stage0_annotate.py --config config.yaml \
    --labels-csv data/annotations/labels.csv \
    --resume
```

**Thứ tự chạy của pipeline:**
1. Whisper ASR → transcript tiếng Việt
2. OpenFace → Action Units từ khuôn mặt
3. Qwen2.5-VL → mô tả visual (tải → xử lý batch → unload)
4. Qwen2-Audio → mô tả âm thanh (tải → xử lý batch → unload)
5. Consolidator (Qwen2.5-7B/32B) → reasoning tổng hợp

Annotations được lưu tại `data/annotations/<clip_id>.json`.

---

## 5. Stage 1–2 — Trích xuất đặc trưng

Các encoder (AST, ViT-FER, ViT-ImageNet, XLM-RoBERTa) được ĐÓNG BĂNG và chỉ dùng để trích xuất features. Features được cache để training nhanh hơn.

```bash
# Trích xuất tất cả modalities
python scripts/extract_features.py --config config.yaml

# Chỉ một số modalities
python scripts/extract_features.py --config config.yaml \
    --modalities audio text

# Bỏ qua các clip đã cache
python scripts/extract_features.py --config config.yaml --skip-existing

# Force tái tính toán
python scripts/extract_features.py --config config.yaml --overwrite
```

Features được lưu tại `data/features/<clip_id>.pt`.

---

## 6. Stage 3–4 — Huấn luyện mô hình (Perception)

Huấn luyện Fusion + Classifier trên features đã cache. Encoders bị đóng băng.

```bash
# Training cơ bản (30 epochs, cosine scheduler, early stopping)
python scripts/train.py --config config.yaml --stage perception

# Resume từ checkpoint
python scripts/train.py --config config.yaml --stage perception \
    --resume-from outputs/checkpoints/perception_best.pt

# Giảm epochs để test nhanh
python scripts/train.py --config config.yaml --stage perception \
    training.perception.epochs=3

# Dùng fusion module khác (ablation)
python scripts/train.py --config config.yaml --stage perception \
    fusion.type=late
```

Checkpoint tốt nhất được lưu tại `outputs/checkpoints/perception_best.pt`.

**Các fusion module có thể dùng:** `conv_attention_4m` (mặc định), `late`, `early`, `mult`, `q_former`, `conv_only`, `attn_only`

---

## 7. Stage 5 — Huấn luyện LLM Reasoner (Cognition + RLVR)

### 7.1 Stage 2 Cognition (LLM + adapter)

```bash
# Cần perception checkpoint từ Stage 1
python scripts/train.py --config config.yaml --stage cognition \
    --resume-from outputs/checkpoints/perception_best.pt
```

### 7.2 LLM-4 RLVR (tùy chọn, nặng)

```bash
# Phase 1: Cold start (SFT ngắn ~2 epochs)
python scripts/train_rlvr.py --config config.yaml --phase cold-start

# Phase 2: GRPO RLVR (cần A100 80GB để dùng Qwen2.5-7B)
python scripts/train_rlvr.py --config config.yaml --phase rlvr \
    --resume-from outputs/checkpoints/llm4_coldstart

# Dùng model nhỏ hơn để test trên Colab
python scripts/train_rlvr.py --config config.yaml --phase cold-start \
    --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 1

# Bật vLLM để generation nhanh hơn trong GRPO
python scripts/train_rlvr.py --config config.yaml --phase rlvr \
    --resume-from outputs/checkpoints/llm4_coldstart --use-vllm
```

---

## 8. Đánh giá

### 8.1 Eval trên test split

```bash
# Eval chuẩn trên test_id
python scripts/eval.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --split test_id

# Eval trên OOD (streamer chưa thấy trong training)
python scripts/eval.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --split test_ood

# Thêm phân tích theo genre
python scripts/eval.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --split test_id --per-genre

# Thêm đánh giá chất lượng reasoning của LLM (chậm hơn)
python scripts/eval.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --include-reasoning
```

### 8.2 Ablation

```bash
# So sánh Strategy A / B / C (Visual encoding strategies)
python scripts/eval.py --config config.yaml --ablation strategy

# So sánh các Fusion modules
python scripts/eval.py --config config.yaml --ablation fusion

# So sánh 4 LLM setups
python scripts/eval.py --config config.yaml --ablation llm
```

Kết quả được lưu tại `outputs/results/eval.json` (hoặc đường dẫn chỉ định bằng `--output`).

---

## 9. Inference và Demo

### 9.1 Inference trên clip mới

```bash
# Một clip
python scripts/infer.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --input path/to/clip.mp4 \
    --output outputs/results/prediction.json

# Một thư mục clips
python scripts/infer.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --input data/new_clips/ \
    --output outputs/results/predictions.json \
    --batch

# Bỏ LLM để chạy nhanh hơn
python scripts/infer.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --input clip.mp4 --no-llm
```

**Output JSON mẫu:**
```json
[
  {
    "clip_id": "clip_001",
    "predicted_label": "hype",
    "confidence": 0.87,
    "class_scores": {"hype": 0.87, "neutral": 0.06, ...},
    "reasoning": "Streamer hét lớn sau khi giết địch cuối...",
    "latency_ms": 245.3
  }
]
```

### 9.2 Gradio Demo

```bash
# Chế độ upload video (batch)
python scripts/demo.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --mode batch --port 7860

# Chế độ realtime (sliding window)
python scripts/demo.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --mode realtime --port 7860

# Chia sẻ qua Gradio public URL
python scripts/demo.py --config config.yaml \
    --checkpoint outputs/checkpoints/perception_best.pt \
    --mode batch --share
```

Mở trình duyệt tại `http://localhost:7860`.

---

## 10. Cấu hình phần cứng (Profiles)

### T4 / L4 16GB (Colab / Kaggle)

```yaml
# config.yaml
compute:
  profile: "colab"

llm:
  base_model:
    name: "Qwen/Qwen2.5-1.5B-Instruct"  # hoặc 7B với 4bit
    quantization: "4bit"

annotation:
  consolidator:
    model_name: "Qwen/Qwen2.5-7B-Instruct"
    quantization: "4bit"
```

Bộ nhớ ước tính:
- Annotation pipeline: ~12 GB VRAM (serial load/unload)
- Training perception: ~4 GB VRAM
- Inference với LLM-1 (Qwen2.5-7B 4bit): ~6 GB VRAM

### A100 80GB (Khuyến nghị cho RLVR)

```yaml
# config.yaml
compute:
  profile: "a100"

llm:
  base_model:
    name: "Qwen/Qwen2.5-7B-Instruct"
    quantization: "none"  # hoặc "4bit" để tiết kiệm VRAM

annotation:
  consolidator:
    model_name: "Qwen/Qwen2.5-72B-Instruct"
    quantization: "4bit"
```

---

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'vie_gameemo'`

```bash
# Thêm src vào PYTHONPATH hoặc cài editable
pip install -e .
# hoặc
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"   # Linux/macOS
$env:PYTHONPATH = "$env:PYTHONPATH;$(pwd)\src"  # Windows PowerShell
```

### `CUDA out of memory`

Giảm batch size hoặc dùng quantization:
```yaml
training:
  perception:
    batch_size: 4          # giảm từ 16
    gradient_accumulation: 16  # tăng để giữ effective batch
llm:
  base_model:
    quantization: "4bit"
```

### `OpenFace binary not found`

Cập nhật đường dẫn trong `config.yaml`:
```yaml
annotation:
  openface:
    binary_path: "/absolute/path/to/OpenFace/build/bin/FeatureExtraction"
```

### Stage 0 bị gián đoạn giữa chừng

Chạy lại với `--resume` — pipeline tự động skip các clip đã có annotation JSON:
```bash
python scripts/stage0_annotate.py --config config.yaml \
    --labels-csv data/annotations/labels.csv --resume
```

### `BitsAndBytesConfig` error trên CPU / macOS

`bitsandbytes` chỉ hỗ trợ CUDA. Tắt quantization:
```yaml
llm:
  base_model:
    quantization: "none"
```

Hoặc thêm flag:
```bash
python scripts/train_rlvr.py ... training.rlvr.quantization=none
```

### Features cache bị lỗi

Xóa cache và trích xuất lại:
```bash
rm -rf data/features/
python scripts/extract_features.py --config config.yaml --overwrite
```

### Logs

Logs được ghi tại `outputs/logs/vie_gameemo.log`. Để debug chi tiết:
```bash
python scripts/train.py --config config.yaml --stage perception \
    logging.level=DEBUG
```

---

## Luồng chạy đầy đủ (tóm tắt)

```
1. Cài đặt môi trường
   pip install torch -r requirements.txt

2. Stage 0 — Thu thập dữ liệu (chạy một lần, nặng)
   stage0_crawl.py → stage0_preprocess.py → stage0_annotate.py

3. Trích xuất features (chạy một lần, cache)
   extract_features.py

4. Huấn luyện Perception (nhanh, lặp nhiều lần cho ablation)
   train.py --stage perception

5. (Tùy chọn) Huấn luyện Cognition
   train.py --stage cognition --resume-from perception_best.pt

6. (Tùy chọn) RLVR
   train_rlvr.py --phase cold-start
   train_rlvr.py --phase rlvr --resume-from llm4_coldstart

7. Đánh giá
   eval.py --checkpoint perception_best.pt --split test_id

8. Demo
   demo.py --checkpoint perception_best.pt --mode batch
```
