# Vie-GameEmo — Mô tả Chi Tiết Pipeline

> Tài liệu này giải thích từng thành phần của hệ thống nhận diện cảm xúc đa phương thức cho livestream game Việt Nam.
> Đọc tuần tự từ Stage 0 → Stage 5 để hiểu luồng dữ liệu đầy đủ.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Stage 0 — Thu thập & Gán nhãn dữ liệu](#2-stage-0--thu-thập--gán-nhãn-dữ-liệu)
3. [Stage 2 — Trích xuất đặc trưng (Encoders)](#3-stage-2--trích-xuất-đặc-trưng-encoders)
4. [Stage 3 — Fusion (Tổng hợp 4 luồng)](#4-stage-3--fusion-tổng-hợp-4-luồng)
5. [Stage 4 — Classifier (Phân loại cảm xúc)](#5-stage-4--classifier-phân-loại-cảm-xúc)
6. [Stage 5 — LLM Reasoning (Giải thích)](#6-stage-5--llm-reasoning-giải-thích)
7. [Huấn luyện (Training)](#7-huấn-luyện-training)
8. [Đánh giá & Suy luận (Eval & Inference)](#8-đánh-giá--suy-luận-eval--inference)
9. [Luồng dữ liệu đầy đủ (Data Flow)](#9-luồng-dữ-liệu-đầy-đủ-data-flow)
10. [Cấu trúc thư mục](#10-cấu-trúc-thư-mục)

---

## 1. Tổng quan kiến trúc

Hệ thống Vie-GameEmo giải quyết bài toán: **Cho một clip video livestream game ~5 giây, xác định cảm xúc của streamer (gaming_8: 8 lớp).**

```
VIDEO CLIP (~5s)
│
├─── AUDIO (wav) ──────── AST Encoder ──────── (B, 64, 768)  ─┐
│                                                              │
├─── FACE CROP (frames) ── FaceViT Encoder ──── (B, 33, 768) ──┤  (16 spatial + 1 global + 16 temporal)
│                                                              ├── Conv-Attention 4M ── (B, T, 768) ── MLP Head ── logits (B, 8)
├─── FULL FRAME (frames) ─ ContextViT Encoder ─ (B, 21, 768) ──┤  (4 spatial + 1 global + 16 temporal)               │
│                                                              │                                              argmax → nhãn
└─── TRANSCRIPT (text) ─── XLM-R/PhoBERT ──── (B, 1, 768)  ─┘
                                                              │
                                                              └── LLM Reasoner ── <think>...</think><answer>nhãn</answer>
```

**5 Stage chính:**

| Stage | Tên | Mục đích |
|-------|-----|---------|
| 0 | Data Collection & Annotation | Thu thập video, gán nhãn cảm xúc tự động |
| 2 | Feature Extraction | Chạy encoders, lưu cache `.pt` |
| 3 | Multimodal Fusion | Gộp 4 luồng thành 1 vector |
| 4 | Classification | Dự đoán nhãn cảm xúc |
| 5 | LLM Reasoning | Giải thích tại sao (4 chiến lược) |

> **Lưu ý đánh số:** Tài liệu gốc dùng Stage 0, 2, 3, 4, 5. Không có Stage 1 — đây là cố ý (Stage 1 là "raw data" trước preprocessing).

---

## 2. Stage 0 — Thu thập & Gán nhãn dữ liệu

### 2a. Crawl video (`src/vie_gameemo/data/crawler.py`)

**Script:** `scripts/stage0_crawl.py`

Dùng `yt-dlp` để tải video từ danh sách URL YouTube. Mỗi video thô sau đó được **cắt thành các clip 5 giây** (`CLIP_TARGET_DURATION=5.0`) bằng ffmpeg.

```
YouTube URL → yt-dlp → video thô (MP4) → ffmpeg segment → clip_001.mp4, clip_002.mp4, ...
```

**Tại sao 5 giây?**  
Quá ngắn → không đủ thông tin cảm xúc. Quá dài → cảm xúc có thể thay đổi. 5 giây là điểm cân bằng: đủ để nhận ra hype/tilted nhưng không gộp nhiều trạng thái.

---

### 2b. Tiền xử lý (`src/vie_gameemo/preprocess/`)

**Script:** `scripts/stage0_preprocess.py`

Mỗi clip trải qua 3 bước:

#### Bước 1 — Demux (`demux.py`)

Tách video thành:
- **Audio:** `ffmpeg` → WAV 16kHz mono → `data/processed/audios/{clip_id}.wav`
- **Frames:** OpenCV trích xuất ở **4fps** → ~20 frame JPG → `data/processed/frames/{clip_id}/frame_0000.jpg`

```
clip.mp4 ──┬── ffmpeg (audio) ──→ clip.wav (16kHz mono)
           └── OpenCV (4fps) ──→ frame_0000.jpg ... frame_0019.jpg
```

> **Tại sao 4fps?** Encoder face/context dùng 16 frame. 5 giây × 4fps = 20 frame → đủ để chọn 16 frame đại diện.

#### Bước 2 — Phát hiện vùng webcam (`webcam_detector.py`)

Trong livestream game, webcam của streamer chiếm ~10-15% màn hình ở góc. Vấn đề: làm sao phân biệt mặt streamer vs mặt nhân vật game (cutscene)?

**Giải pháp: DBSCAN clustering theo vị trí không gian:**

```
Lấy 30 frame mẫu
    ↓
MediaPipe FaceDetection → danh sách bbox (xmin, ymin, w, h) chuẩn hoá [0,1]
    ↓
Tính tâm (cx, cy) từng bbox → DBSCAN(eps=0.05, min_samples=5)
    ↓
Chọn cluster lớn nhất (stable_score = cluster_size/n_sampled > 0.5)
    ↓
Webcam bbox = trung bình cộng bbox trong cluster thắng
```

**Tại sao DBSCAN hoạt động?**
- Webcam ở góc cố định suốt clip → tạo cluster dày đặc ở góc
- Mặt NPC xuất hiện lẻ tẻ ở giữa màn hình → DBSCAN phân loại là noise (-1)
- `edge_bias`: webcam thường gần cạnh màn hình → ưu tiên cluster gần cạnh

#### Bước 3 — Crop mặt (`face_crop.py`)

Dùng `WebcamBBox` từ bước 2 để crop vùng mặt từ mỗi frame, với margin 20%. Kết quả lưu vào `data/processed/faces/{clip_id}/`.

Nếu không phát hiện webcam → `has_face = False` → face encoder nhận tensor toàn 0.

---

### 2c. Pipeline gán nhãn đa tác tử (`src/vie_gameemo/data/annotator/pipeline.py`)

**Script:** `scripts/stage0_annotate.py`

Đây là bước quan trọng nhất của Stage 0. Thay vì gán nhãn thủ công 600 clip (tốn thời gian), hệ thống dùng **4 agent AI tuần tự**, mỗi agent chuyên một phương thức:

```
clip.mp4 + nhãn thủ công → [Phase 1] → [Phase 2] → [Phase 3] → [Phase 4] → annotation.json
```

**Thiết kế quan trọng: Load → Xử lý toàn bộ batch → Unload**

Mỗi GPU (T4: 16GB VRAM) chỉ đủ cho một model lớn tại một thời điểm. Pipeline load model, chạy cho toàn bộ batch, rồi unload trước khi load model tiếp theo.

#### Phase 1 — Xử lý local CPU/GPU (nhẹ)

Không cần LLM lớn. Chạy tuần tự cho từng clip:

**OpenFace AU Extraction** (`openface_au.py`):
- Gọi binary `OpenFace` qua subprocess
- Output: chuỗi thời gian AU intensity theo frame (AU1, AU4, AU6, AU12, AU17, AU23, AU24, v.v.)
- AU (Action Units) là các đơn vị hành động cơ mặt theo FACS. Ví dụ: AU12 = cơ kéo góc miệng lên (cười), AU4 = cơ nhíu mày.

**Peak Frame Detection** (`peak_frame.py`):
- Tìm frame có AU intensity tổng hợp cao nhất → frame thể hiện cảm xúc mạnh nhất
- Dùng weighted sum của AU intensity, weight theo độ liên quan cảm xúc
- Fallback: frame giữa clip nếu OpenFace không chạy được

**Webcam Detection + ASR:**
- `WebcamDetector` xác định vùng webcam
- `WhisperASR` hoặc `PhoWhisperASR` (cấu hình trong config.yaml) transcribe audio → text tiếng Việt
- Optional: `BARTphoPostProcessor` làm sạch ASR output (sửa lỗi chính tả, hoàn thiện câu)

#### Phase 2 — Qwen2.5-VL: Mô tả hình ảnh (`qwen_vl_agent.py`)

Load `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit quantization, ~6GB VRAM):

**Input:** Peak frame JPG  
**Prompt (tiếng Việt):** "Mô tả khách quan biểu cảm khuôn mặt, tư thế cơ thể, và bối cảnh game trong ảnh. Đừng suy diễn cảm xúc, chỉ mô tả những gì thấy."  
**Output:** Chuỗi mô tả khách quan → `visual_objective_desc`

Ví dụ output: *"Người chơi há miệng, hai tay giơ lên, màn hình hiển thị bảng kill streak 10 liên tiếp."*

#### Phase 3 — Qwen2-Audio: Mô tả âm thanh (`qwen_audio_agent.py`)

Load `Qwen/Qwen2-Audio-7B-Instruct`:

**Input:** WAV audio 16kHz  
**Output:** Mô tả prosody + âm thanh game → `audio_tone_desc`

Ví dụ output: *"Giọng nói to và nhanh, có tiếng hô lớn ở giây thứ 3, âm game chiến thắng ở nền."*

#### Phase 4 — Consolidator: Reasoning tổng hợp (`consolidator.py`)

Load `Qwen/Qwen2.5-7B-Instruct` (có thể là 32B/72B nếu có A100):

**Input:** Gộp tất cả thông tin:
- Nhãn cảm xúc thủ công (`emotion_label`)
- AU intensities từ OpenFace (`face_aus`)
- Mô tả hình ảnh từ Qwen-VL (`visual_objective_desc`)
- Mô tả âm thanh từ Qwen-Audio (`audio_tone_desc`)
- Transcript từ Whisper (`transcript`)

**Output:** Reasoning text giải thích tại sao nhãn này phù hợp → `reasoning`

Ví dụ: *"Clip này nhãn HYPE vì: (1) AU12 cao liên tục → cười to, (2) Qwen-VL mô tả tay giơ lên, (3) Qwen-Audio phát hiện giọng to nhanh và hô lớn, (4) Transcript 'BOOM! ACE!' xác nhận."*

#### Output cuối: `Annotation` JSON

```json
{
  "clip_id": "streamer1_clip_042",
  "emotion_label": "hype",
  "genre": "fps",
  "face_aus": {"AU12": 2.8, "AU25": 3.1, ...},
  "peak_frame_idx": 14,
  "webcam_bbox": {"xmin": 0.02, "ymin": 0.70, "width": 0.22, "height": 0.28, ...},
  "visual_objective_desc": "Người chơi há miệng...",
  "audio_tone_desc": "Giọng nói to và nhanh...",
  "transcript": "BOOM! ACE!",
  "reasoning": "Clip này nhãn HYPE vì...",
  "annotators": [...],
  "human_verified": false,
  "created_at": "2026-05-29T..."
}
```

---

## 3. Stage 2 — Trích xuất đặc trưng (Encoders)

**Script:** `scripts/extract_features.py`

Sau khi có annotation JSON và frames/audio, chạy 4 encoder để tạo **feature cache** (file `.pt`). Training đọc từ cache này thay vì chạy lại encoder mỗi epoch → nhanh hơn hàng chục lần.

```
annotations/ + frames/ + audios/
        ↓ (extract_features.py)
data/features/
    clip_001.pt   → {'audio': (64,768), 'face': (1,768), 'context': (1,768), 'text': (1,768)}
    clip_002.pt
    ...
```

### Encoder 1 — Audio: AST (`encoders/audio_ast.py`)

**Model:** `MIT/ast-finetuned-audioset-10-10-0.4593`

AST (Audio Spectrogram Transformer) xử lý audio theo cách tương tự ViT xử lý ảnh:
1. Tạo log-mel spectrogram từ audio 16kHz
2. Chia spectrogram thành các patch (như ViT)
3. Transformer encoder trên các patch → sequence of token vectors
4. Adaptive average pooling → **output shape: `(1, 64, 768)`**

```
audio.wav (16kHz) → log-mel spectrogram → AST patches → Transformer → pool → (1, 64, 768)
```

**Tại sao AST thay vì MFCC truyền thống?**
AST được pre-train trên AudioSet (2M clips, 527 event classes). Nó đã học phân biệt tiếng cười, tiếng hét, tiếng khóc — các âm thanh liên quan trực tiếp đến cảm xúc game.

**T=64** vì: clip 5 giây → AST tạo khoảng 1200 patch → pool xuống 64 token để phù hợp với sequence length của audio. Đây là chiều dài chuỗi dài nhất trong 4 modality, dùng làm target khi align.

---

### Encoder 2 — Face: ViT-FER (`encoders/face_vit.py`)

**Model:** `trpakov/vit-face-expression`

ViT pre-train trên **AffectNet** (1M ảnh khuôn mặt có nhãn cảm xúc). Thiết kế **tri-view**:

```
face crops (N frames)
    ├── Spatial view:  peak (middle) frame → patch tokens → spatial pool (4×4) → 16 tokens
    ├── Global view:   peak frame → CLS token                                  →  1 token
    └── Temporal view: 16 evenly-sampled frames, each CLS token kept           → 16 tokens
                                                                                 ─────────
                                           concat [patches | global_CLS | temporal_CLS] → (1, 33, 768)
```

**Ba view bổ sung nhau:**
- **Spatial patches:** Chi tiết cục bộ (micro-expression: nếp nhăn khóe miệng, nâng mày)
- **Global CLS:** Tổng quan biểu cảm tại khoảnh khắc đỉnh cảm xúc
- **Temporal CLS:** Chuỗi thời gian — cười → hét → cười trong 5 giây

**pool_method** cho spatial patches: `"mean"` (avg pool), `"max"` (max pool), `"attention"` (CLS-guided weighted pool — highlight vùng liên quan cảm xúc không cần extra parameters).

**QUAN TRỌNG:** Encoder này nhận **face crop** (vùng webcam đã crop), KHÔNG phải full frame. Nếu `has_face=False` (không tìm thấy webcam) → trả về tensor toàn 0.

---

### Encoder 3 — Context: ViT-B/16 (`encoders/context_vit.py`)

**Model:** `google/vit-base-patch16-224`

ViT standard được pre-train trên ImageNet-21k. Xử lý webcam region crop (hoặc full frame khi không tìm thấy webcam), nắm bắt bối cảnh: body language, tư thế, và game screen. Thiết kế **tri-view** giống FaceEncoder:

```
webcam/full frames (N frames)
    ├── Spatial view:  peak (middle) frame → patch tokens → spatial pool (2×2) →  4 tokens
    ├── Global view:   peak frame → CLS token                                  →  1 token
    └── Temporal view: 16 evenly-sampled frames, each CLS token kept           → 16 tokens
                                                                                 ─────────
                                           concat [patches | global_CLS | temporal_CLS] → (1, 21, 768)
```

**Tại sao tri-view cho context?**
- **Spatial patches:** Chi tiết UI cục bộ (kill feed góc trên, health bar dưới) không visible trong CLS
- **Global CLS:** Scene semantics tổng thể (combat vs menu vs cutscene)
- **Temporal CLS:** Thay đổi cảnh theo thời gian (calm → sudden combat → aftermath) — không bị mất khi mean-pool

---

### Encoder 4 — Text: XLM-R hoặc PhoBERT (`encoders/text_xlmr.py`, `text_phobert.py`)

Chọn qua `config.yaml` → `text_encoder.type: "xlmr"` hoặc `"phobert"`:

**XLM-RoBERTa** (`FacebookAI/xlm-roberta-base`):
- Multilingual (100 ngôn ngữ)
- **Tốt cho:** code-switching Việt-Anh (streamer mix "ACE rồi bro!", "skill đẹp quá")
- Output: CLS token → `(1, 1, 768)`

**PhoBERT** (`vinai/phobert-base-v2`):
- Monolingual Vietnamese (20GB text tiếng Việt)
- **Tốt cho:** transcript tiếng Việt thuần (ít gaming slang Anh)
- Output: CLS token → `(1, 1, 768)`

```yaml
# config.yaml
text_encoder:
  type: "xlmr"       # hoặc "phobert"
  xlmr:
    model_name: "FacebookAI/xlm-roberta-base"
  phobert:
    model_name: "vinai/phobert-base-v2"
```

Factory `build_text_encoder(cfg.text_encoder)` tự chọn đúng class theo config.

---

### Ý nghĩa 3 chiều của output tensor `(B, T, D)`

Mọi encoder đều trả về tensor 3 chiều. Đây là quy ước chung của toàn bộ pipeline:

```
tensor shape: (B, T, D)
               │  │  │
               │  │  └── D = Feature dimension = 768
               │  └────── T = Sequence length (số "token" theo thời gian)
               └────────── B = Batch size (số clip xử lý song song)
```

#### Chiều 1 — B (Batch size)

Khi training, nhiều clip được xử lý song song trên GPU để tăng tốc. `B=16` nghĩa là 16 clip đang được xử lý cùng một lúc.

Khi inference một clip đơn lẻ, `B=1`.

Encoder luôn giữ nguyên chiều này và trả về đúng B samples.

---

#### Chiều 2 — T (Sequence length / số token thời gian)

Đây là chiều quan trọng nhất, có **ý nghĩa khác nhau tùy modality**:

**Audio — T = 64**

AST xử lý spectrogram bằng cách chia thành các patch nhỏ theo trục tần số × thời gian, rồi cho qua Transformer. Mỗi patch trở thành một "token". Clip 5 giây ở 16kHz tạo ra ~1200 patch — quá nhiều cho downstream model xử lý. Pipeline pool xuống còn **64 token**, mỗi token đại diện cho ~78ms audio.

```
5 giây audio → ~1200 AST patches → pool → 64 token
                                              ↑
                              mỗi token = đặc trưng của ~78ms âm thanh
                              (cười, hét, im lặng, tiếng game,...)
```

Giữ T=64 (không pool thành 1) vì cảm xúc audio thay đổi theo thời gian: giây đầu im lặng, giây thứ 3 hét to → thông tin temporal quan trọng.

**Face — T = 33** (với spatial_pool=(4,4), n_temporal=16)

Tri-view FaceEncoder giữ nguyên 3 loại token: 16 patch tokens (spatial) + 1 CLS (global) + 16 CLS (temporal). Lý do giữ T=33 thay vì pool thành T=1:
- **Spatial patches** (T=16): biểu cảm vi mô — nếp nhăn, cơ mắt — không thể hiện qua CLS đơn thuần
- **Global CLS** (T=1): biểu cảm tổng thể tại peak frame
- **Temporal CLS** (T=16): diễn biến cảm xúc theo thời gian (chuỗi được Fusion Conv branch xử lý)
- Fusion module aligns 33 → T_target=64 bằng interpolation

**Context — T = 21** (với spatial_pool=(2,2), n_temporal=16)

Tri-view ContextEncoder tương tự: 4 patch tokens (spatial) + 1 CLS (global) + 16 CLS (temporal). Lý do không dùng mean-pool-to-1 nữa:
- Mean pool từ 16 CLS về 1 vector **mất toàn bộ thông tin temporal** — clip có thể bắt đầu bình yên rồi đột ngột combat, thông tin đó biến mất khi average
- Spatial patches nắm chi tiết UI (kill feed, health bar) không visible trong CLS
- Fusion handles T=21 → 64 giống như face và audio

**Text — T = 1**

XLM-R tokenize transcript thành WordPiece tokens, Transformer xử lý, lấy CLS token (vị trí đặc biệt đầu chuỗi) → 1 vector đại diện **toàn bộ câu nói**. Đủ cho nhiệm vụ sentence-level classification. (Nếu `pooling='none'` thì T = số token, nhưng mặc định là 'cls' → T=1.)

---

#### Chiều 3 — D (Feature dimension = 768)

Tất cả encoder đều output 768 chiều — đây là kích thước embedding của BERT-base và các model cùng họ. Được giữ cố định để:
1. Mọi modality có cùng "ngôn ngữ" số học → dễ tổng hợp trong Fusion
2. 768 là điểm cân bằng: đủ phong phú để biểu diễn cảm xúc, không quá lớn gây overfit trên 600 clip

Mỗi số trong vector 768 chiều không có ý nghĩa riêng lẻ — đây là không gian học được. Vector gần nhau trong không gian 768D → cảm xúc tương tự.

---

### Tóm tắt output shapes

| Modality | Encoder | Output `(B, T, D)` | B | T | D |
|----------|---------|-------------------|---|---|---|
| Audio | AST | `(B, 64, 768)` | batch | 64 token thời gian (~78ms/token) | 768 features |
| Face | ViT-FER | `(B, 33, 768)` | batch | 16 spatial patches + 1 global CLS + 16 temporal CLS | 768 features |
| Context | ViT-B/16 | `(B, 21, 768)` | batch | 4 spatial patches + 1 global CLS + 16 temporal CLS | 768 features |
| Text | XLM-R/PhoBERT | `(B, 1, 768)` | batch | 1 CLS token toàn câu | 768 features |

> **Lưu ý ký hiệu:** Trong code, encoder trả về `(1, T, 768)` khi encode một clip đơn lẻ (B=1). Khi đọc từ file `.pt` và xếp thành batch, collate_fn gộp thành `(B, T, 768)`. Trong tài liệu spec, đôi khi ghi `(B, T, D)` với D=768 hiểu ngầm.

Tất cả đều có D = **768** → Fusion nhận vào 4 tensor cùng "ngôn ngữ số học", chỉ khác ở T.

---

## 4. Stage 3 — Fusion (Tổng hợp 4 luồng)

Đây là trung tâm của hệ thống. Nhiệm vụ: nhận 4 tensor `(B, T_i, 768)` → ra `(B, T, 768)` tổng hợp.

### Module chính: Conv-Attention 4M (`fusion/conv_attention.py`)

**Thiết kế lấy cảm hứng từ Emotion-LLaMAv2 (2026)**, được điều chỉnh cho 4 modality.

#### Bước 1 — MLP Standardization

Mỗi modality qua một Linear layer riêng:
```
audio → mlp_audio → u_a  (B, T_a, 768)
face  → mlp_face  → u_f  (B, T_f, 768)
ctx   → mlp_ctx   → u_c  (B, T_c, 768)
text  → mlp_text  → u_t  (B, T_t, 768)
```

Mục đích: "dịch" từng modality về cùng một không gian embedding trước khi gộp.

#### Bước 2 — Align sequences to target length T

Vấn đề: T_audio=64, T_face=1, T_context=1, T_text=1 → không thể concat trực tiếp.

**Giải pháp:** Align tất cả về T = T_audio = 64:
- Nếu T_i == 1 → **broadcast** (expand) → `(B, 64, 768)` (nhanh, không mất thông tin)
- Nếu T_i > 1 và ≠ 64 → **linear interpolation** (`F.interpolate`)
- Nếu T_i == T → giữ nguyên

#### Bước 3 — Tạo hai cấu trúc hybrid

```python
F_d = torch.cat([u_a, u_f, u_c, u_t], dim=-1)    # (B, T, 4*768=3072) — ngang
F_s = torch.stack([u_a, u_f, u_c, u_t], dim=-1)  # (B, T, 768, 4)     — xếp chồng
```

- `F_d` (dense): concatenate theo chiều feature → dùng cho Conv branch và tính attention weights
- `F_s` (stacked): giữ nguyên từng modality riêng → dùng để tính weighted sum

#### Bước 4 — Hai nhánh song song

---

##### Nhánh Conv — Bắt pattern thay đổi THEO THỜI GIAN

**Câu hỏi nhánh này trả lời:** "Cảm xúc thay đổi như thế nào trong 5 giây này?"

Streamer có thể im lặng 3 giây rồi đột ngột hét to — đây là pattern temporal. Conv1d hoạt động như một "cửa sổ trượt" nhìn vào 3 timestep liền kề và phát hiện pattern đó.

```
F_d (B, 64, 3072)
    │
    ↓ Conv1d(3072 → 768, kernel=3, padding=1)
    │   Mỗi output token = nhìn token [t-1, t, t+1] của tất cả 4 modality
    │   → compress từ 3072 → 768 channel
    │
    ↓ ResidualConvBlock × 4
    │   Mỗi block:
    │     Conv1d(768→768, k=3) → SwitchAct → Conv1d(768→768, k=3) + residual
    │   → tinh chỉnh pattern ngày càng phức tạp hơn
    │
    ↓ F_conv (B, 64, 768)
```

**Conv1d hoạt động thế nào?**

```
Token 0: [audio₀, face₀, ctx₀, text₀]  ─┐
Token 1: [audio₁, face₁, ctx₁, text₁]  ─┼─ Conv kernel nhìn 3 token liền kề
Token 2: [audio₂, face₂, ctx₂, text₂]  ─┘    → output token 1: pattern tại t=1
```

Kernel size=3 nghĩa là mỗi output token "nhìn" 3 input token liền kề (±1 bước thời gian = ±78ms). Sau 4 ResidualConvBlock, mỗi token có receptive field lên đến ±4 bước = ±312ms — đủ để bắt sự chuyển đổi cảm xúc trong clip 5 giây.

**Residual connection** giúp gradient không bị triệt tiêu khi backprop qua 4 block (vấn đề vanishing gradient):
```
output = Conv(input) + input    ← cộng trực tiếp input vào output
```

**SwitchActivation** `= x * sigmoid(x)`:
- Gần 0 → gate gần đóng → đặc trưng yếu bị suppress
- Lớn dương → gate mở → đặc trưng mạnh được khuếch đại
- Khác ReLU: vẫn cho gradient nhỏ về phía âm → học tốt hơn

---

##### Nhánh Attention — Chọn MODALITY NÀO QUAN TRỌNG tại mỗi thời điểm

**Câu hỏi nhánh này trả lời:** "Ở khoảnh khắc này, nên tin vào giọng nói, mặt, cảnh game, hay lời thoại hơn?"

Tùy tình huống, tầm quan trọng của mỗi modality rất khác nhau:

| Tình huống | Audio | Face | Context | Text |
|-----------|-------|------|---------|------|
| Streamer đang nói "ACE!" | cao | trung bình | thấp | cao |
| Streamer im lặng nhìn màn hình | thấp | cao | cao | thấp |
| Clip không có webcam (`has_face=False`) | cao | 0 (zeroed) | cao | trung bình |

Nhánh này học **tự động** những trọng số đó — không cần hard-code.

```
F_d (B, 64, 3072)
    │
    ↓ MLP: Linear(3072→1536) → GELU → Linear(1536→4)
    │   Nhìn vào tất cả 4 modality tại timestep t
    │   → ra 4 điểm số raw (1 cho mỗi modality)
    │
    ↓ Softmax trên 4 điểm → attention weights (B, 64, 4)
    │   weights[b, t, :] = [w_audio, w_face, w_ctx, w_text]
    │   tổng = 1.0, mỗi weight ∈ (0, 1)
    │
    ↓ Weighted sum trên F_s
    │   F_s shape: (B, 64, 768, 4)  ← 4 modality stack theo dim cuối
    │   output[b, t, d] = Σ_m weights[b, t, m] * F_s[b, t, d, m]
    │
    ↓ F_attn (B, 64, 768)
```

**Ví dụ attention weights tại 3 timestep khác nhau trong cùng 1 clip:**
```
t=5  (im lặng):  audio=0.12  face=0.51  ctx=0.30  text=0.07
t=20 (đang nói): audio=0.55  face=0.25  ctx=0.08  text=0.12
t=55 (hét to):   audio=0.70  face=0.18  ctx=0.04  text=0.08
```

Những weights này không được đặt trước — chúng được **học** trong quá trình training thông qua backpropagation.

---

#### Bước 5 — Kết hợp hai nhánh

```python
u_fusion = F_conv + F_attn    # (B, 64, 768)
```

Cộng trực tiếp — không có trọng số thêm. Hai nhánh đóng góp **vai trò bổ sung nhau**:

```
F_conv:  "Giọng tăng dần rồi đột ngột to ở t=40"  ← thông tin temporal
F_attn:  "Tại t=40, tin vào audio 70%"             ← thông tin selection
u_fusion = cả hai gộp lại = "Âm thanh to đột ngột lúc t=40 là dấu hiệu chính"
```

---

### Ablation study — Tại sao cần cả hai nhánh?

Ablation study là thí nghiệm **tháo từng phần ra** để đo xem phần đó đóng góp bao nhiêu. Có 6 variant được implement để so sánh với Conv-Attention 4M đầy đủ:

#### Nhóm 1 — Ablation hai nhánh nội bộ

**`conv_only`** — Tháo bỏ AttentionBranch, chỉ giữ ConvBranch:
```
F_d → ConvBranch → F_conv → Classifier
```
Kết quả kỳ vọng: kém hơn Conv-Attention 4M vì không biết chọn modality.
Nếu `conv_only` gần bằng Conv-Attention 4M → AttentionBranch đóng góp ít → cần xem lại.

**`attn_only`** — Tháo bỏ ConvBranch, chỉ giữ AttentionBranch:
```
F_d, F_s → AttentionBranch → F_attn → Classifier
```
Kết quả kỳ vọng: kém hơn vì không bắt được temporal pattern.
Nếu `attn_only` gần bằng Conv-Attention 4M → ConvBranch đóng góp ít → cần xem lại.

So sánh 3 dòng này cho thấy đóng góp riêng của từng nhánh:
```
Conv-Attention 4M:  Macro F1 = X%   (mục tiêu đạt cao nhất)
conv_only:          Macro F1 = X-a% (a = đóng góp của Attention nhánh)
attn_only:          Macro F1 = X-b% (b = đóng góp của Conv nhánh)
```

#### Nhóm 2 — So sánh với các phương pháp fusion khác trong văn献

**`late` (Late Fusion)** — Baseline đơn giản nhất, gộp sau khi pool:
```
Mỗi modality → mean pool → (B, 768) → Linear → average
→ output (B, 1, 768)
```
Không có bất kỳ tương tác nào giữa các modality trước khi gộp. Mất toàn bộ thông tin temporal. Điểm chuẩn thấp nhất kỳ vọng.

**`early` (Early Fusion)** — Gộp ngay từ đầu:
```
[mean_audio, mean_face, mean_ctx, mean_text] → concat (B, 3072) → MLP → (B, 768)
```
Có tương tác giữa các modality nhưng sau khi đã pool mất temporal. Hơn `late` một chút vì MLP học được cross-modal interaction.

**`mult` (Multimodal Transformer / MULT)** — Từ paper Tsai et al. ACL 2019:
```
Mỗi modality cross-attend với audio làm Key/Value:
  Q = modality_i, K = V = audio → cross-attended_i
→ concat 4 output → project → (B, T_audio, 768)
```
Cải tiến lớn: có temporal structure và cross-modal attention. Hạn chế: audio làm anchor cứng — nếu audio nhiễu thì toàn bộ hệ thống bị ảnh hưởng. Không có Conv branch để bắt local pattern.

**`q_former` (Q-Former / AffectGPT style)** — 32 learnable query tokens cross-attend với toàn bộ modality tokens:
```
32 query tokens (học được) cross-attend với [audio; face; ctx; text] → (B, 32, 768)
```
Linh hoạt hơn MULT — queries có thể attend bất kỳ modality nào. Nhưng không có temporal conv → không bắt pattern thời gian ngắn hạn.

#### Kết quả kỳ vọng sau ablation

```
Rank | Model            | Macro F1 | Lý do
-----|------------------|----------|-------
1    | Conv-Attention4M | cao nhất | Cả hai nhánh + 4 modality đầy đủ
2    | q_former         | tốt      | Cross-modal flexible nhưng thiếu conv
3    | mult             | tốt      | Có temporal nhưng audio-anchored
4    | conv_only        | trung bình | Temporal OK nhưng không chọn modality
5    | attn_only        | trung bình | Selection OK nhưng thiếu temporal
6    | early            | thấp     | Mất temporal, pool quá sớm
7    | late             | thấp nhất | Không có cross-modal interaction
```

Nếu kết quả thực tế không đúng thứ tự này → cần điều tra lại (ví dụ: nếu `late` tốt hơn `conv_only` → ConvBranch đang bị training không ổn định).

---

## 5. Stage 4 — Classifier (Phân loại cảm xúc)

**File:** `src/vie_gameemo/classifiers/mlp.py`

Thiết kế đơn giản có chủ ý — không cần phức tạp vì Fusion đã làm phần khó.

```python
EmotionClassifier(d_model=768, hidden_dim=256, n_classes=8, dropout=0.3, pool='mean')
```

**Forward pass:**

```
u_fusion (B, T, 768)
    ↓ pool (mean/max/cls token) → (B, 768)
    ↓ Linear(768→256) + GELU + Dropout(0.3)
    ↓ Linear(256→8)
    → logits (B, 8)
```

**Tại sao GELU thay vì ReLU?**  
GELU (Gaussian Error Linear Unit) có gradient mượt hơn, hoạt động tốt hơn cho các embedding từ Transformer.

**Tại sao dropout=0.3?**  
Dataset nhỏ (~600 clip), dropout mạnh hơn để tránh overfit.

**3 chế độ pooling:**
- `mean`: trung bình tất cả T tokens → ổn định nhất
- `max`: lấy max mỗi chiều → bắt peak feature mạnh
- `cls`: lấy token đầu tiên → dùng khi fusion thêm CLS token

---

## 6. Stage 5 — LLM Reasoning (Giải thích)

Có **4 chiến lược** (setup) với input và cơ chế khác nhau. Bảng so sánh nhanh:

| Setup | Model | Training | Input nguồn | Nhãn đến từ đâu | Annotation-free? |
|-------|-------|----------|------------|----------------|-----------------|
| LLM-1 | Qwen2.5-7B | ❌ Không | Text features từ annotation | Classifier cho sẵn | ❌ |
| LLM-2 | Qwen2.5-7B | ✅ SFT (LoRA) + ModalAdapter | Text evidence **hoặc** fusion_emb | LLM tự predict | ✅ (cần cognition ckpt) |
| LLM-3 | Qwen2.5-VL-7B | ✅ LoRA | Raw frames + audio | LLM tự predict | ✅ Luôn luôn |
| LLM-4 | Qwen2.5-7B | ✅ SFT → GRPO + ModalAdapter | Text evidence **hoặc** fusion_emb | LLM tự predict | ✅ (cần cognition ckpt) |

---

### Modal Adapter — Bridge Embedding → LLM (annotation-free)

**File:** `src/vie_gameemo/llm/modal_adapter.py`

**Vấn đề:** Khi inference trên clip mới (không có annotation JSON), LLM-2/4 không có `face_aus`,
`visual_objective`, `audio_tone` để xây dựng text prompt → trả về "N/A" evidence, reasoning vô nghĩa.

**Giải pháp:** **Modal Adapter** — linear projection d_fusion → d_llm, convert tensor 768-dim thành
"soft tokens" được inject trực tiếp vào LLM embedding space (cơ chế từ Emotion-LLaMAv2, Section 4.4):

```
u_fusion: (1, T, 768)        ← output từ ConvAttention4M (Stage 3)
    ↓ ModalAdapter [Linear 768 → 4096]
    ↓ mean pool over T
soft_token: (1, 1, 4096)     ← 1 soft token trong LLM embedding space

instruction: "Phân tích cảm xúc từ đặc trưng đa phương thức..."
    ↓ tokenize → embed_tokens
text_embeds: (1, L, 4096)

inputs_embeds = cat([soft_token, text_embeds], dim=1)  # (1, 1+L, 4096)
    ↓ model.generate(inputs_embeds=inputs_embeds)       # NO input_ids
<think>[lý luận từ soft token]</think><answer>LABEL</answer>
```

**Soft token ≠ text** — LLM không nhìn thấy chữ mà nhìn thấy vector số 4096-dim ngay trong embedding
space của nó. LLM có thể attend vào đó qua self-attention như token thường.

**Dispatch logic** (trong `llm2_coreasoner.py` và `llm4_rlvr.py`):
```python
def reason(self, evidence: dict) -> LLMOutput:
    if "fusion_emb" in evidence and self.modal_adapter is not None:
        return self._reason_with_embeddings(evidence["fusion_emb"])  # soft token path
    return self._reason_with_text(evidence)                          # text path (annotation có sẵn)
```

**Batch inference** (`batch.py`) tự động chọn path:
```python
# _forward() luôn trả về fusion_emb
prediction = _forward(fusion, classifier, features, device)  # gồm "fusion_emb" key

# _features_to_evidence() chọn path phù hợp
if not has_annotation and fusion_emb is not None:
    return {"fusion_emb": fusion_emb, "label": ...}   # → modal adapter path
else:
    return {"face_aus": ..., "visual_objective": ...}  # → text evidence path
```

**Training:** `ModalAdapter` được train cùng LLM LoRA trong Stage 2 (Cognition).
Saved tại `outputs/checkpoints/cognition_best.pt["llm_adapter"]`.

**Config để enable:**
```yaml
llm:
  cognition_checkpoint: "outputs/checkpoints/cognition_best.pt"
```

---

### LLM-1: Post-hoc Explainer (`llm/llm1_explainer.py`)

**Không train.** Dùng model off-the-shelf với prompt engineering.

**Nguồn input — từ hai nơi khác nhau:**

```
┌─ Stage 4 output ───────────────────────────────────────────────────┐
│  MLP Classifier → predicted label = "hype"                         │
│  (đây là kết quả đã chạy qua AST→Fusion→Classifier)               │
└────────────────────────────────────────────────────────────────────┘
         +
┌─ Stage 0 output (Annotation JSON) ─────────────────────────────────┐
│  face_aus:     {"AU12": 2.8, "AU25": 3.1, "AU4": 0.3, ...}        │  ← OpenFace
│  transcript:   "BOOM! ACE rồi bro!"                                │  ← Whisper
│  game_context: "Người chơi há miệng, tay giơ lên..."               │  ← Qwen-VL
│  pitch_hz/rms_db/shout: prosody features                            │  ← Qwen-Audio
└────────────────────────────────────────────────────────────────────┘
```

**Hai cách đưa thông tin vào LLM-1:**

1. **Text evidence path** (khi có annotation): Stage 0 agent AI viết sẵn mô tả ngôn ngữ tự nhiên về âm thanh và hình ảnh → LLM đọc được trực tiếp.

2. **Không có annotation**: LLM-1 không có fallback — vẫn cần text evidence. Chỉ LLM-2/4 mới hỗ trợ annotation-free qua ModalAdapter (xem phần Modal Adapter ở đầu section này).

Prompt được build từ hai nguồn trên:
```
[USER]
  Nhãn classifier: hype
  AU: AU12=2.8, AU25=3.1, AU4=0.3
  Transcript: "BOOM! ACE rồi bro!"
  Bối cảnh game: Người chơi há miệng, tay giơ lên...
  Giọng nói: to, nhanh, hô lớn ở giây 3
  → Hãy giải thích tại sao nhãn này đúng.
```

**Output:**
```
<think>AU12 cao → cười to. Transcript "ACE" xác nhận chiến thắng.
Giọng to nhanh → hype.</think>
<answer>hype</answer>
```

**Điểm yếu quan trọng:** LLM nhận nhãn từ classifier như sự thật đã cho trước → chỉ viết lý do bênh vực nhãn đó. Nếu classifier sai, LLM vẫn giải thích sai một cách tự tin. Đây là "post-hoc rationalization", không phải reasoning thật sự.

---

### LLM-2: Co-Reasoner (`llm/llm2_coreasoner.py`)

**SFT bằng LoRA.** LLM đóng vai trò **suy luận + phân loại**, thay thế hoàn toàn MLP Classifier trong inference.

**Nguồn input — chỉ từ Stage 0 (Annotation JSON), KHÔNG có nhãn classifier:**

```
┌─ Stage 0 output (Annotation JSON) ─────────────────────────────────┐
│  face_aus:         {"AU12": 2.8, "AU25": 3.1, ...}  ← OpenFace    │
│  visual_objective: "Người chơi há miệng, tay giơ lên..." ← Qwen-VL│
│  audio_tone:       "Giọng nói to, nhanh, hô lớn..." ← Qwen-Audio  │
│  transcript:       "BOOM! ACE rồi bro!"             ← Whisper      │
└────────────────────────────────────────────────────────────────────┘
```

Prompt không có `label` field — LLM phải tự suy ra:
```
[USER]
  Khuôn mặt (AU): AU12=2.8, AU25=3.1, ...
  Bối cảnh game: Người chơi há miệng, tay giơ lên...
  Giọng nói: to, nhanh, hô lớn...
  Transcript: "BOOM! ACE rồi bro!"
  Nhãn có thể: neutral, hype, amused, tilted, sad, shocked, fear, disgusted
  → Phân tích và dự đoán cảm xúc.
```

**MLP Classifier có chạy không?**

| Giai đoạn | MLP Classifier | LLM-2 |
|-----------|---------------|-------|
| **Cognition training** | ✅ Chạy (FROZEN) — chỉ để cung cấp `L_cls` | ✅ Train (LoRA) |
| **Inference** | ❌ Không cần chạy | ✅ Chạy, output `<answer>` là nhãn cuối |

Trong **cognition training**, Classifier chạy nhưng không phải để LLM dùng — chỉ để tính `L_cls` ngăn representation drift. LLM-2 được train song song bằng `L_reasoning_LM` (next-token prediction trên reasoning text từ annotation).

Trong **inference**, MLP Classifier bị bypass hoàn toàn. Chỉ cần annotation JSON text → LLM-2 → `<answer>`. Luồng encoder→fusion→classifier không được gọi.

**Training target (supervision signal từ annotation JSON):**
```
<think>Clip này nhãn HYPE vì AU12 cao → cười to, Qwen-VL mô tả
tay giơ lên, audio hô lớn, transcript ACE xác nhận.</think>
<answer>hype</answer>
```
Đây là `reasoning` field do Consolidator (Stage 0 Phase 4) viết — LLM-2 học bắt chước cách suy luận đó.

---

### LLM-3: VLM End-to-End (`llm/llm3_vlm.py`)

**LoRA fine-tune `Qwen2.5-VL-7B-Instruct`** — VLM nhận ảnh trực tiếp, không dùng text features từ annotation.

**Input — raw frames và audio, KHÔNG qua AST/FaceViT/XLM-R:**

```
frame_paths: [frame_0002.jpg, frame_0006.jpg, ..., frame_0018.jpg]  ← 8 frames sample đều
             (lấy từ data/processed/frames/{clip_id}/)

audio_path:  data/processed/audios/{clip_id}.wav                    ← optional
             → render thành mel spectrogram PNG → đưa vào như ảnh thứ 9

transcript:  "BOOM! ACE rồi bro!"                                   ← text đính kèm
```

**VLM nhìn thấy:**
```
[IMG: frame_0002] [IMG: frame_0006] ... [IMG: frame_0018] [IMG: spectrogram.png]
Text: "BOOM! ACE rồi bro! → Phân tích cảm xúc streamer."
```

**Spectrogram image trick:**
```
audio.wav → librosa.melspectrogram → matplotlib figure → PNG 224×224
```
VLM vốn không xử lý audio, nhưng mel spectrogram là ảnh — VLM có thể "nhìn" hình dạng tần số của âm thanh (vùng năng lượng cao = giọng to, pattern đột ngột = hét).

**So sánh với LLM-1/2:**
- LLM-1/2 nhận **text mô tả** về hình ảnh (do Qwen-VL viết) → thông tin đi qua bottleneck ngôn ngữ
- LLM-3 nhận **hình ảnh gốc** → có thể thấy biểu cảm mà text mô tả không nắm hết (micro-expression, eye contact)
- Nhược điểm: không dùng AST encoder chuyên âm thanh → spectrogram image kém AST về thông tin audio

---

### LLM-4: RLVR (`llm/llm4_rlvr.py`)

**Input — giống LLM-2** (text features, không có nhãn sẵn). Khác biệt nằm ở **cách training**, không phải input.

```
face_aus, transcript, visual_objective_desc, audio_tone_desc
    ↓ (giống LLM-2)
Qwen2.5-7B → sinh ra completions
    ↓ (khác LLM-2: không dùng teacher forcing)
Reward functions đánh giá completions → GRPO update
```

**Hai giai đoạn training:**

**Giai đoạn 1 — Cold start SFT** (50-100 sample tốt nhất):
LLM học format `<think>...</think><answer>...</answer>` trước. Nếu bỏ bước này, GRPO sẽ mất nhiều iteration chỉ để LLM học cú pháp output.

**Giai đoạn 2 — GRPO:**
```
Cho mỗi training sample:
  1. Sinh G=8 completions khác nhau (temperature sampling)
  2. Tính reward mỗi completion:
       R_acc    = 1.0  nếu <answer> == ground_truth_label
                  0.0  nếu sai hoặc không parse được
       R_format = 1.0  nếu có đúng <think>...</think><answer>...</answer>
                  0.0  nếu thiếu tag
       R = R_acc + R_format  →  ∈ {0.0, 1.0, 2.0}
  3. Tính advantage = R_i - mean(R_1..G)
     (completion tốt hơn trung bình nhóm → advantage dương)
  4. Cập nhật policy: tăng xác suất completion có advantage > 0
                       giảm xác suất completion có advantage < 0
```

**Tại sao GRPO tốt hơn SFT (LLM-2)?**

LLM-2 học bắt chước reasoning của Consolidator (teacher forcing) — kể cả khi reasoning đó không hoàn toàn đúng. GRPO chỉ quan tâm một điều: **cuối cùng có predict đúng nhãn không**. LLM tự khám phá cách reasoning nào dẫn đến câu trả lời đúng → robust hơn với OOD data.

---

### Format output chung cho cả 4 setup

```
<think>
[Phân tích 3-5 câu: AU nào cao, transcript nói gì, visual mô tả gì, audio thế nào]
→ Kết luận cảm xúc là X.
</think>
<answer>neutral|hype|amused|tilted|sad|shocked|fear|disgusted</answer>
```

Parser (`LLM1Explainer.parse_output`) dùng regex trích xuất `<answer>` tag. Nếu không match → `format_valid=False` → fallback về nhãn classifier.

---

## 7. Huấn luyện (Training)

**Script:** `scripts/train.py`

### Tổng quan: Thành phần nào được train, thành phần nào bị freeze?

```
┌─────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Thành phần      │ Stage 0 (annotation) │ Perception training  │ Cognition training   │
├─────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ AST (Audio)     │ FROZEN (pretrained)  │ FROZEN + cached .pt  │ FROZEN + cached .pt  │
│ FaceViT         │ FROZEN (pretrained)  │ FROZEN + cached .pt  │ FROZEN + cached .pt  │
│ ContextViT      │ FROZEN (pretrained)  │ FROZEN + cached .pt  │ FROZEN + cached .pt  │
│ XLM-R/PhoBERT   │ FROZEN (pretrained)  │ FROZEN + cached .pt  │ FROZEN + cached .pt  │
│ Fusion module   │ không dùng           │ ✅ TRAIN (LR=2e-4)   │ FROZEN               │
│ MLP Classifier  │ không dùng           │ ✅ TRAIN (LR=2e-4)   │ FROZEN               │
│ ModalAdapter    │ không dùng           │ không dùng           │ ✅ TRAIN (LR=2e-4)   │
│ LLM (Qwen2.5)   │ FROZEN (inference)   │ không dùng           │ ✅ TRAIN LoRA (2e-5) │
└─────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

**Quy tắc chung:** Mỗi giai đoạn chỉ train phần mới thêm vào, giữ nguyên phần đã train trước. Lý do:
- Encoder đã học tốt từ hàng triệu sample → không cần và không nên fine-tune trên 600 clip
- Khi train LLM, Fusion+Classifier đã ổn định → không để gradient từ LLM "phá" các weights đó
- Cách tiếp cận này gọi là **staged/modular training** — phổ biến trong các hệ thống multimodal lớn

---

### Perception Training (`training/perception.py`)

**Chỉ train Fusion + Classifier. Encoders FROZEN, features đọc từ cache.**

```
data/features/clip_001.pt  ──┐
data/features/clip_002.pt  ──┤ DataLoader đọc .pt trực tiếp
...                          ──┘  (không chạy encoder)
     ↓
  Fusion (TRAIN)  →  u_fusion (B, T, 768)
     ↓
  Classifier (TRAIN)  →  logits (B, 8)
     ↓
  Loss → backprop → cập nhật weights Fusion + Classifier
```

**Tại sao freeze encoder và cache features?**
1. Encoder forward pass tốn 80-90% thời gian một iteration — cache giúp training nhanh ~50×
2. 600 clip quá ít để fine-tune encoder 86M param (AST) mà không overfit
3. Encoder đã pretrain trên AudioSet/AffectNet/ImageNet — những domain này đủ gần với game streaming

**Hyperparameters:**
- Optimizer: AdamW, weight_decay=0.01
- LR: 2e-4 (cho cả Fusion lẫn Classifier)
- Epochs: 30, early stopping patience=5 theo `val_macro_f1`
- Batch size: 16, gradient accumulation nếu VRAM nhỏ
- Mixed precision: fp16 (`GradScaler`)
- Warmup: 10% đầu tổng steps (lr tăng tuyến tính từ 0 → 2e-4)

---

### Hàm Loss được dùng và thử nghiệm

**File:** `training/losses.py` — implement hai hàm loss:

#### Focal Loss (mặc định, dùng trong Perception)

```
FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

  p_t  = softmax probability của class đúng
  γ    = 2.0   (focusing parameter)
  α    = 1.0   (class weight — đặt 1.0, không dùng per-class weight)
```

**Tại sao cần Focal Loss?**

Dataset gaming_8 mất cân bằng: neutral chiếm 20%, disgusted chỉ 7%. Nếu dùng Cross-Entropy thông thường, model học thiên về predicting neutral.

Focal Loss hoạt động bằng cách **nhân thêm hệ số `(1-p_t)^γ`**:
- Sample dễ (neutral, model đã dự đoán đúng với p=0.9): hệ số = `(1-0.9)^2 = 0.01` → loss nhỏ → gradient nhỏ → model không "lãng phí" capacity vào sample này
- Sample khó (fear, model dự đoán sai với p=0.3): hệ số = `(1-0.3)^2 = 0.49` → loss lớn → gradient lớn → model buộc phải học class khó này

**Khi γ=0:** Focal Loss = Cross-Entropy thông thường → có thể dùng γ=0 làm ablation để đo tác dụng.

#### Weighted Cross-Entropy (thay thế, dùng cho ablation loss)

```
WCE = -Σ_c  w_c * y_c * log(p_c)

  w_c = 1 / class_frequency_c   (inverse frequency weighting)
```

Class hiếm (fear, disgusted) có weight cao hơn → loss của chúng được khuếch đại.

**Khi nào dùng WCE thay Focal Loss?** Khi dataset có class rất hiếm (< 3%) — WCE stable hơn Focal trong trường hợp đó.

**Thử nghiệm loss (ablation):**

| Config | Loss | Ghi chú |
|--------|------|---------|
| Mặc định | Focal (γ=2, α=1) | Khuyến nghị |
| Ablation A | Cross-Entropy (γ=0) | Focal vs CE thuần |
| Ablation B | Focal γ=1 | Ít focus hơn |
| Ablation C | Focal γ=3 | Focus cực mạnh vào hard samples |
| Ablation D | Weighted CE | Thay thế cho dataset cực mất cân bằng |

Chạy ablation bằng cách thay `cfg.classifier.loss.focal.gamma` trong config.yaml.

---

### Cognition Training (`training/cognition.py`)

**FREEZE Fusion + Classifier. Chỉ train LLM qua LoRA.**

```
data/features/clip_001.pt → Fusion (FROZEN) → Classifier (FROZEN) → logits (B, 8)
                                                                           ↓
annotation.json → reasoning text ──────────────────────────────── LLM (TRAIN LoRA)
                                                                           ↓
                                                              L_cls + L_reasoning
```

**Loss tổng hợp Cognition:**
```
L_total = α * L_cls + β * L_reasoning_LM

  L_cls          = FocalLoss(logits, labels)          ← giữ Classifier không drift
  L_reasoning_LM = CrossEntropy(LLM_logits, tokens)  ← LLM học viết reasoning
  α = 1.0,  β = 0.5
```

`L_reasoning_LM` là **language modeling loss** chuẩn: cho LLM xem reasoning text từ Stage 0, tính loss dự đoán token tiếp theo. LLM học cách viết reasoning kiểu `"AU12 cao → cười, transcript 'ACE' → HYPE"`.

`L_cls` được giữ để tránh Classifier drift trong quá trình backprop từ LLM.

**LoRA config:**
- rank=16, alpha=32 → effective learning rate scale = alpha/rank = 2× → ~1-2% số param được train
- Target modules: `q_proj, v_proj` (attention matrices của LLM)
- LR: 2e-5 (10× nhỏ hơn Perception — LLM đã pretrained tốt, chỉ cần tinh chỉnh nhẹ)

---

### VieGameEmoDataset (`data/dataset.py`)

Hai chế độ:
- **`cached` mode** (mặc định, dùng cho training): Đọc `.pt` từ `data/features/` → fast
- **`raw` mode** (inference): Chạy encoder on-the-fly → chậm nhưng không cần pre-extract

**Collate function** (`collate_fn`): Vì T có thể khác nhau giữa các clip trong batch (nếu raw mode), padding zero đến max T rồi stack.

**Split manifest:** JSON file `splits.json` ánh xạ `clip_id → train/val/test_id/test_ood`. Stratified split theo emotion label đảm bảo mỗi split có đủ đại diện các class.

---

## 8. Đánh giá & Suy luận (Eval & Inference)

### Metrics (`evaluation/metrics.py`)

| Metric | Ý nghĩa |
|--------|---------|
| **Accuracy** | % clip dự đoán đúng |
| **Macro F1** | F1 trung bình không weighted → quan trọng với class nhỏ (fear, disgusted) |
| **Weighted F1** | F1 theo tỉ lệ class → bias về class lớn (neutral) |
| **UAR** | Unweighted Average Recall = Macro Recall — tiêu chuẩn trong SER |
| **Per-class F1** | F1 riêng mỗi trong 8 nhãn |
| **Confusion matrix** | 8×8 ma trận nhầm lẫn |

**Metric chính để so sánh:** `Macro F1` và `UAR` (vì dataset mất cân bằng).

### Per-genre Evaluation (`evaluation/per_genre.py`)

Tính metrics riêng cho từng genre game (FPS, MOBA, Horror, RPG, Casual, Mobile). Quan trọng để phát hiện bias:
- Model có thể tốt với MOBA (nhiều data) nhưng kém với Horror (data hiếm)
- Fear/disgusted phổ biến hơn trong Horror game

### Strategy Ablation (`evaluation/strategy_ablation.py`)

So sánh 3 chiến lược inference:
- **Strategy A:** Chỉ classifier (không LLM)
- **Strategy B:** LLM-1 post-hoc (classifier + explain)
- **Strategy C:** LLM-4 RLVR (LLM tự predict)

### Batch Inference (`inference/batch.py`)

Với clip mới (không có annotation sẵn), pipeline tự chạy encoder inline:

```
clip_mới.mp4
    ↓ extract_audio() + extract_frames()
    ↓ ASTAudioEncoder + FaceEncoder + ContextEncoder + XLMRTextEncoder
    → features dict (audio, face, context, text tensors)
    ↓ Fusion → Classifier → predicted_label + confidence
    ↓ (optional) LLM
```

**Vấn đề khi kết hợp LLM-1/2 với clip mới:**

`_features_to_evidence()` (dòng 250-258 trong `batch.py`) cố build evidence dict từ features:
```python
return {
    "face_aus":         features.get("face_aus", "N/A"),           # → "N/A"
    "visual_objective": features.get("visual_description", "N/A"), # → "N/A"
    "audio_tone":       features.get("audio_description", "N/A"),  # → "N/A"
    "transcript":       features.get("transcript", ""),            # → ""
    "label":            features.get("predicted_label", "neutral"),
}
```

Với clip mới, features dict **chỉ chứa tensor 768-dim** (audio, face, context, text) — không có text descriptions. Kết quả: LLM-1/2 nhận hầu hết "N/A", chỉ biết `label` từ classifier → reasoning rất nghèo nàn.

**Giải pháp cho từng LLM setup với clip mới:**

| Setup | Clip trong training set | Clip hoàn toàn mới |
|-------|------------------------|-------------------|
| **MLP Classifier** | ✅ Dùng cache `.pt` | ✅ Chạy encoder inline |
| **LLM-1** | ✅ Dùng annotation JSON | ⚠️ Phải chạy Whisper + OpenFace + Qwen-VL trước |
| **LLM-2** | ✅ Dùng annotation JSON | ⚠️ Phải chạy Whisper + OpenFace + Qwen-VL trước |
| **LLM-3** | ✅ Dùng raw frames | ✅ Chạy trực tiếp trên frames — không cần annotation |
| **LLM-4** | ✅ Dùng annotation JSON | ⚠️ Phải chạy Whisper + OpenFace + Qwen-VL trước |

→ **LLM-3 là setup duy nhất hoạt động hoàn toàn tự lập** với clip mới, vì nó nhận raw frames thay vì text từ annotation.

---

### Real-time Inference (`inference/realtime.py`)

**Sliding window** 5 giây, stride 1 giây. Thực tế trong code có một trade-off rất thực tế:

```python
# _process_window() trong RealtimeInferenceRunner
face_tensor, has_face = self._encode_faces(pil_frames)   # ✅ chạy FaceEncoder
ctx_tensor = self._encode_context(pil_frames)             # ✅ chạy ContextEncoder
audio_tensor = self._zero_audio()                         # ❌ tensor toàn 0
text_tensor  = self._zero_text()                          # ❌ tensor toàn 0
```

**Audio và text bị zero ra hoàn toàn** trong real-time mode. Lý do: ASR (Whisper) cần encode toàn bộ audio clip — không thể làm streaming real-time trong <600ms. Tương tự XLM-R cần transcript đã có sẵn.

Vì vậy real-time inference thực chất chỉ dùng **2/4 modality** (face + context), còn audio và text đóng góp bằng 0.

```
Live stream
    ↓ buffer 5 giây frames
    ↓ FaceEncoder(frames) → (1, 1, 768)       ← có dữ liệu
    ↓ ContextEncoder(frames) → (1, 1, 768)    ← có dữ liệu
    ↓ zeros(1, 64, 768)                        ← audio giả (constraint real-time)
    ↓ zeros(1, 1, 768)                         ← text giả
    ↓ _compute_fused() → u_fusion (1, T, 768) ← stored cho LLM
    ↓ _predict_from_fused() → label
```

**LLM bị skip mặc định** (`skip_llm=True`) — quá chậm cho real-time loop.

**On-demand LLM cho "highlight moment":** Khi phát hiện một window đáng chú ý, gọi `explain_window(window_id)` để trigger LLM asynchronously:

```
Real-time loop: window_0 → window_1 → window_2 [HYPE detected!] → window_3 → ...
                                                       ↓ (background thread)
                                               explain_window(2)
                                                   ↓ lấy fusion_emb từ window buffer
                                                   ↓ {"fusion_emb": tensor} → LLM-2/4
                                                   ↓ ModalAdapter → soft token
                                                   ↓ LLM → reasoning (annotation-free)
```

**ModalAdapter giải quyết gap real-time**: window result giờ lưu `fusion_emb` tensor. `explain_window()` truyền nó như evidence `{"fusion_emb": ...}` → LLM-2/4 tự động dùng ModalAdapter path, không cần annotation JSON.

---

## 9. Luồng dữ liệu đầy đủ (Data Flow)

```
YouTube livestream
    │
    ▼ stage0_crawl.py (yt-dlp)
video thô (.mp4)
    │
    ▼ stage0_preprocess.py (ffmpeg + OpenCV)
data/clips/{clip_id}.mp4        ← clip 5 giây
data/processed/audios/{id}.wav  ← 16kHz mono
data/processed/frames/{id}/*.jpg ← 4fps ~20 frames
data/processed/faces/{id}/*.jpg  ← face crops
    │
    ▼ stage0_annotate.py (4-phase multi-agent)
data/annotations/{clip_id}.json ← Annotation với emotion_label + reasoning
    │
    ▼ extract_features.py (4 encoders, frozen)
data/features/{clip_id}.pt ← {'audio':(64,768), 'face':(1,768), 'context':(1,768), 'text':(1,768)}
    │
    ▼ train.py --stage perception
checkpoints/best_perception.pt ← fusion + classifier weights
    │
    ▼ train.py --stage cognition (optional)
checkpoints/best_cognition.pt ← LLM LoRA adapter weights
    │
    ▼ eval.py / infer.py / demo.py
kết quả dự đoán + reasoning
```

---

## 10. Cấu trúc thư mục

```
vie-gameemo-skeleton/
│
├── config.yaml                     ← Tất cả hyperparameters & paths
├── INSTRUCTIONS.md                 ← Hướng dẫn chạy nhanh
│
├── src/vie_gameemo/
│   ├── data/
│   │   ├── schemas.py              ← EmotionLabel enum, Annotation, MultimodalFeatures
│   │   ├── dataset.py              ← VieGameEmoDataset + collate_fn
│   │   ├── crawler.py              ← yt-dlp wrapper
│   │   ├── feature_cache.py        ← batch encode → lưu .pt
│   │   └── annotator/
│   │       ├── pipeline.py         ← annotate_batch() 4-phase orchestrator
│   │       ├── openface_au.py      ← OpenFace subprocess wrapper
│   │       ├── peak_frame.py       ← AU-based peak frame detection
│   │       ├── whisper_asr.py      ← WhisperASR + PhoWhisperASR + BARTpho + build_asr()
│   │       ├── qwen_vl_agent.py    ← Qwen2.5-VL visual description
│   │       ├── qwen_audio_agent.py ← Qwen2-Audio prosody description
│   │       └── consolidator.py     ← Qwen2.5-Instruct reasoning consolidation
│   │
│   ├── preprocess/
│   │   ├── demux.py                ← extract_audio() + extract_frames()
│   │   ├── webcam_detector.py      ← MediaPipe + DBSCAN webcam localization
│   │   └── face_crop.py            ← Crop webcam region với margin
│   │
│   ├── encoders/
│   │   ├── audio_ast.py            ← ASTAudioEncoder → (B, 64, 768)
│   │   ├── face_vit.py             ← FaceEncoder (dual-view) → (B, 1, 768)
│   │   ├── context_vit.py          ← ContextEncoder → (B, 1, 768)
│   │   ├── text_xlmr.py            ← XLMRTextEncoder + build_text_encoder()
│   │   └── text_phobert.py         ← PhoBERTTextEncoder
│   │
│   ├── fusion/
│   │   ├── __init__.py             ← register_fusion() decorator + get_fusion()
│   │   ├── conv_attention.py       ← ConvAttention4M (recommended)
│   │   └── baselines.py            ← Late, Early, MULT, Q-Former, conv_only, attn_only
│   │
│   ├── classifiers/
│   │   └── mlp.py                  ← EmotionClassifier (768→256→8)
│   │
│   ├── training/
│   │   ├── losses.py               ← FocalLoss + WeightedCE
│   │   ├── perception.py           ← train_perception() + evaluate()
│   │   └── cognition.py            ← train_cognition() (joint cls+reasoning)
│   │
│   ├── llm/
│   │   ├── base.py                 ← BaseLLMReasoner ABC + LLMOutput dataclass
│   │   ├── llm1_explainer.py       ← Post-hoc, no training
│   │   ├── llm2_coreasoner.py      ← SFT co-reasoner
│   │   ├── llm3_vlm.py             ← VLM LoRA (Qwen2.5-VL)
│   │   └── llm4_rlvr.py            ← GRPO + reward functions
│   │
│   ├── evaluation/
│   │   ├── metrics.py              ← accuracy, F1, UAR, confusion matrix
│   │   ├── per_genre.py            ← breakdown by game genre
│   │   ├── strategy_ablation.py    ← A/B/C strategy comparison
│   │   └── reasoning_eval.py       ← đánh giá chất lượng reasoning text
│   │
│   ├── inference/
│   │   ├── batch.py                ← batch inference pipeline
│   │   └── realtime.py             ← sliding window real-time
│   │
│   └── utils/
│       ├── config.py               ← load_config() YAML + CLI overrides
│       ├── logging.py              ← setup_logging()
│       ├── seed.py                 ← set_seed() determinism
│       └── io.py                   ← read/write JSON/YAML, file_hash
│
├── scripts/
│   ├── stage0_crawl.py
│   ├── stage0_preprocess.py
│   ├── stage0_annotate.py
│   ├── extract_features.py
│   ├── train.py
│   ├── train_rlvr.py
│   ├── eval.py
│   ├── infer.py
│   └── demo.py
│
├── notebooks/
│   ├── 01_kaggle_stage0_prepare_data.ipynb  ← Crawl + Preprocess + Annotate (Kaggle T4)
│   ├── 02_kaggle_training.ipynb             ← Feature extract + Perception training
│   └── 03_kaggle_inference.ipynb            ← Load model + Demo inference
│
└── docs/
    ├── annotation_guideline.md              ← Định nghĩa 8 nhãn + quyết định mơ hồ
    ├── labeling_manual.md                   ← Hướng dẫn cho người gán nhãn thủ công
    └── pipeline_overview.md                 ← Tài liệu này
```

---

## Tóm tắt các quyết định thiết kế quan trọng

| Quyết định | Lý do |
|-----------|-------|
| **Clip 5 giây** | Đủ ngắn để cảm xúc đồng nhất, đủ dài cho AST/ViT hoạt động |
| **4fps extraction** | ~20 frames/clip → đủ để sample 16 frames cho dual-view face encoder |
| **DBSCAN cho webcam** | Phân biệt mặt streamer (stable) vs mặt NPC (sporadic) mà không cần labeled data |
| **Load→Batch→Unload** | T4 GPU 16GB VRAM: không thể giữ nhiều LLM cùng lúc |
| **Feature caching** | Training nhanh hơn ~50x; encoder không cần gradient trong perception stage |
| **768 cho mọi modality** | Dùng chung BERT-style dimension → dễ concat và fuse |
| **Conv + Attention song song** | Conv bắt temporal patterns; Attention chọn modality → bổ sung nhau |
| **Focal Loss** | Dataset mất cân bằng: neutral 20%, disgusted 7% → FL tập trung class khó |
| **LoRA cho LLM** | Chỉ train ~1% param → tránh catastrophic forgetting, tiết kiệm VRAM |
| **GRPO thay vì RLHF** | Reward verifiable (đúng/sai nhãn) → không cần reward model riêng |
