# Vie-GameEmo — Pipeline Chạy (Inference)

Mô tả luồng dữ liệu từ video clip thô → nhãn cảm xúc + giải thích.  
Không bao gồm quá trình training hay annotation.

---

## Dataset & Kết quả Training

### Phân bố dataset (gaming_8, 3399 clip ~5s)

| Nhãn | Train | Val | Test | Tổng | % |
|------|------:|----:|-----:|-----:|--:|
| neutral | 993 | 213 | 212 | 1418 | 41.7% |
| amused | 296 | 64 | 62 | 422 | 12.4% |
| hype | 243 | 52 | 46 | 341 | 10.0% |
| fear | 230 | 49 | 49 | 328 | 9.6% |
| sad | 197 | 41 | 43 | 281 | 8.3% |
| tilted | 200 | 43 | 37 | 280 | 8.2% |
| shocked | 156 | 34 | 32 | 222 | 6.5% |
| disgusted | 74 | 17 | 16 | 107 | 3.1% |
| **TOTAL** | **2389** | **513** | **497** | **3399** | 100% |

Xử lý class imbalance khi training :
1. Effective Number Class Weights
2. Focal Loss (γ=2.0) + class weights làm α
3. Balanced Batch Sampler

Dataset mất cân bằng nặng: `neutral` chiếm 41.7%, `disgusted` chỉ 3.1%.

### Xử lý class imbalance khi training

Ba kỹ thuật kết hợp, tất cả đều trong `training/perception.py` + `training/losses.py`:

**1. Effective Number Class Weights**

Thay vì dùng `1/count` đơn thuần (quá nhạy với class cực hiếm), dùng công thức *effective number of samples*:

```
β = (N − 1) / N
effective_num_c = 1 − β^(count_c)
weight_c = (1 − β) / effective_num_c
```

Kết quả normalize (mean=1) từ training log:
```
neutral:   0.221   ← class phổ biến nhất → weight thấp
hype:      0.645
amused:    0.777
tilted:    0.936
sad:       0.950
fear:      0.819
shocked:   1.189
disgusted: 2.464   ← class hiếm nhất → weight cao nhất
```

**2. Focal Loss (γ=2.0) + class weights làm α**

```
loss = α_c · (1 − p_t)^γ · CE(logits, label)
```

- `α_c` = effective_number weight của class c → tăng penalty cho class hiếm
- `(1 − p_t)^γ` với γ=2: nếu model đã confident đúng (p_t→1) → loss ≈ 0; nếu khó/sai (p_t nhỏ) → loss giữ nguyên → tập trung vào hard examples

**3. Balanced Batch Sampler (WeightedRandomSampler)**

```python
sample_weight[i] = 1.0 / count_of_class(label[i])
WeightedRandomSampler(sample_weights, replacement=True)
```

Mỗi batch được sample với xác suất tỉ lệ nghịch với tần suất class → class hiếm (`disgusted`, `shocked`) xuất hiện gần như thường xuyên như `neutral` trong mỗi batch.

**4. Checkpoint selection theo Macro F1 (không phải accuracy)**

val_macro_f1 tính trung bình F1 không weight → một model chỉ predict `neutral` sẽ bị phạt nặng, không được lưu làm best checkpoint.

---

### Kết quả training — Perception (Conv-Attention 4M + MLP)

**Best checkpoint:** epoch 89 | `outputs/checkpoints/perception_best.pt`

#### Val (epoch 89 — best val)

| Metric | Giá trị |
|--------|--------:|
| val_accuracy | **85.6%** |
| val_macro_f1 | **0.8169** |
| val_UAR | **0.8299** |

Per-class val:

| Nhãn | F1 | Recall |
|------|----|--------|
| neutral | 0.933 | 0.907 |
| sad | 0.929 | 0.951 |
| fear | 0.880 | 0.936 |
| amused | 0.857 | 0.831 |
| disgusted | 0.857 | 0.882 |
| hype | 0.748 | 0.816 |
| tilted | 0.700 | 0.700 |
| shocked | 0.632 | 0.615 |

#### Test set (497 mẫu)

| Metric | Giá trị |
|--------|--------:|
| Accuracy | **0.8330** |
| Macro F1 | **0.7724** |
| UAR | **0.7799** |

Per-class test:

| Nhãn | F1 | Recall | Precision | Support |
|------|----|--------|-----------|--------:|
| sad | 0.953 | 0.953 | 0.953 | 43 |
| neutral | 0.929 | 0.901 | 0.960 | 212 |
| amused | 0.883 | 0.855 | 0.914 | 62 |
| hype | 0.762 | 0.696 | 0.842 | 46 |
| fear | 0.722 | 0.714 | 0.729 | 49 |
| tilted | 0.660 | **0.892** | 0.524 | 37 |
| disgusted | 0.645 | 0.625 | 0.667 | 16 |
| shocked | 0.585 | 0.594 | 0.576 | 32 |

**Top confusion pairs (test):**

| Predict sai | → Predict thành | Số lỗi | Tỉ lệ |
|-------------|-----------------|-------:|------:|
| hype | → tilted | 8 | 17.4% |
| neutral | → tilted | 7 | 3.3% |
| amused | → tilted | 6 | 9.7% |
| fear | → tilted | 5 | 10.2% |
| neutral | → shocked | 4 | 1.9% |

`tilted` là class bị predict nhầm vào nhiều nhất từ các class khác — recall cao (0.892) nhưng precision thấp (0.524), cho thấy model over-predict `tilted`.

---

## Tổng quan

```
video clip (~5s)
    │
    ├── 1. Tiền xử lý
    │       ├── ffmpeg       → WAV 16kHz + JPG frames 4fps
    │       ├── YOLO + DBSCAN → webcam bbox
    │       ├── MediaPipe    → face crop (trong webcam bbox)
    │       └── Whisper (faster-whisper) → transcript text
    │
    ├── 2. Trích xuất đặc trưng (4 encoder, tất cả frozen)
    │       ├── WAV          → Whisper encoder-only  → audio_emb  (1, 64, 768)
    │       ├── face_crop    → ViT-FER (tri-view)    → face_emb   (1, 33, 768)
    │       ├── webcam_crop  → ViT-B/16              → ctx_emb    (1,  1, 768)
    │       └── transcript   → CafeBERT              → text_emb   (1,  1, 768)
    │
    ├── 3. Fusion — Conv-Attention
    │       │
    │       │  Align về T=64: face/ctx/text (T=1) → expand; audio (T=64) → giữ nguyên
    │       │
    │       │  F_d = concat([audio, face, ctx, text], dim=-1)  → (1, 64, 3072)
    │       │  F_s = stack( [audio, face, ctx, text], dim=-1)  → (1, 64, 768, 4)
    │       │
    │       ├── Nhánh Conv  — bắt thay đổi theo thời gian:
    │       │       F_d → Conv1d(3072→768, k=3) → ResConvBlock×4 → F_conv (1, 64, 768)
    │       │
    │       ├── Nhánh Attention — chọn modality quan trọng tại mỗi timestep:
    │       │       F_d → MLP(3072→1536→4) → Softmax → weights (1, 64, 4)
    │       │       weights × F_s → F_attn (1, 64, 768)
    │       │
    │       └── u_fusion = F_conv + F_attn  →  (1, 64, 768)
    │
    ├── 4. Classifier
           └── MLP: u_fusion → mean pool → Linear(768→256)+GELU+Dropout → Linear(256→8) → logits → nhãn
                                                                       
    │
    └── 5. LLM
            │
            ├── 5a. ModalAdapter — project 5 luồng → LLM space (d=3584)
            │       ├── u_fusion   (1, 64, 768) → proj_fusion  → (1, 64, 3584)
            │       ├── penult     (1,  1, 256) → proj_penult  → (1,  1, 3584)
            │       ├── audio_emb  (1, 64, 768) → proj_audio   → (1, 64, 3584)
            │       ├── face_emb   (1, 33, 768) → proj_face    → (1, 33, 3584)
            │       └── ctx_emb    (1,  1, 768) → proj_context → (1,  1, 3584)
            │               ↓ cat theo seq dim
            │       soft_tokens: (1, 163, 3584)
            │       layout: [penult | fusion×64 | audio×64 | face×33 | ctx×1]
            │
            ├── 5b. Build prompt
            │     transcript → _CUE_PROMPT → text_embeds (1, L_text, 3584)
            │
            ├── 5c. inputs_embeds = cat([soft_tokens, text_embeds], dim=1)
            │              → (1, 163+L_text, 3584)
            │
            └── 5d. Qwen2.5-7B-Instruct.generate(inputs_embeds=...)
                    → "Cues: face: ...; voice: ...; scene: ...; text: .... Emotion: nhãn."
```

---

## Bước 1 — Tiền xử lý

**Input:** `clip.mp4` (~5 giây)

### 1a. Tách audio

```
ffmpeg -i clip.mp4 -ar 16000 -ac 1 clip.wav
```

Output: `clip.wav` — mono 16kHz, ~80 000 mẫu cho 5 giây.

### 1b. Tách frames

```
OpenCV: đọc video ở 4fps → frame_0000.jpg ... frame_0019.jpg (~20 frames)
```

Pipeline dùng 16 frames trong số đó cho encoder.

### 1c. Phát hiện vùng webcam

Webcam của streamer chiếm góc nhỏ (~10–15%) màn hình. Cần tách mặt streamer ra khỏi mặt nhân vật game.

```
30 frame mẫu
    │
    ▼ YOLO → danh sách bbox khuôn mặt (xmin, ymin, w, h) chuẩn hoá [0,1]
    │
    ▼ DBSCAN(eps=0.08, min_samples=3) theo tâm (cx, cy) của mỗi bbox
    │   webcam ở góc cố định → cluster dày
    │   mặt NPC ở giữa màn hình → bị phân loại là noise
    │
    ▼ Chọn cluster lớn nhất gần cạnh màn hình
    │
    └── WebcamBBox (bbox vùng webcam của streamer)
             │
             ▼ Crop từng frame theo WebcamBBox (+ margin 20%)
             │       → webcam_crop: vùng chứa cả mặt + thân trên streamer
             │
             ▼ MediaPipe FaceDetection (chạy trong webcam_crop)
             │       → face_bbox (tọa độ trong webcam_crop)
             │
             └── face_crop (224×224) → ViT-FER
                 webcam_crop         → ViT-B/16 (context)
```

Nếu không tìm được webcam → `has_face = False` → face encoder nhận tensor toàn 0.  
Nếu tìm được webcam nhưng MediaPipe không detect được mặt → face encoder nhận tensor toàn 0; context encoder vẫn dùng `webcam_crop`.

### 1d. ASR — Chuyển audio → text transcript

```
clip.wav
    │
    ▼ faster-whisper (openai/whisper-large-v3)
    │   language=None → Whisper tự phát hiện ngôn ngữ (routing="auto")
    │   initial_prompt chứa gaming terms VI + EN → cải thiện độ chính xác
    │   VAD filter lọc đoạn im lặng (threshold=0.4)
    │   nếu language_probability < 0.6 → fallback về "vi"
    │   hallucination filter loại bỏ "subscribe", "đăng ký", URL...
    │
    ▼ fastText LID cross-check (lid.176.ftz) — chỉ để audit
    │   phát hiện ngôn ngữ thực của transcript, không override
    │
    └── transcript: "BOOM! ACE rồi bro!"
        asr_detected_language: "vi"
```

---

## Bước 2 — Trích xuất đặc trưng

Bốn encoder chạy **độc lập**, tất cả **frozen** (không cập nhật trọng số khi inference). Output đều có D=768.

### 2a. Audio Encoder — Whisper encoder-only

**Model:** `openai/whisper-small` (244M, encoder-only — không có decoder)

Chỉ dùng phần encoder của Whisper để trích xuất đặc trưng prosody (ngữ điệu, nhịp nói, độ nhấn mạnh). Khác với Bước 1d dùng `whisper-large-v3` để ra text — ở đây chỉ lấy hidden states, không decode ra chữ.

```
clip.wav (16kHz)
    │
    ▼ librosa.load() → waveform numpy array
    │
    ▼ AutoFeatureExtractor → log-mel spectrogram (input_features)
    │
    ▼ WhisperModel.encoder (frozen, no decoder)
    │   last_hidden_state: (1, N, 768)
    │   (nếu model hidden ≠ 768 → Linear projection → 768)
    │
    ▼ adaptive_avg_pool1d → 64 token
    │
    └── audio_feat: (1, 64, 768)
        mỗi token ≈ 78ms âm thanh
```

### 2b. Face Encoder — ViT-FER (tri-view)

**Model:** `trpakov/vit-face-expression` (pretrain AffectNet)

Input là `face_crop` từ **MediaPipe FaceDetection chạy trong webcam_crop** (Bước 1c). Không nhận full frame.

```
face_crops từ MediaPipe (đã crop trong webcam region)
    │
    ├── Spatial view:  frame giữa → ViT → patch tokens → pool 4×4 (16, 768)
    ├── Global view:   frame giữa → ViT → CLS token         (1, 768)
    └── Temporal view: 16 frames → ViT per frame → CLS     (16, 768)
            ↓
        cat([spatial | global_CLS | temporal_CLS])
            ↓
        face_feat: (1, 33, 768)   (16 patch + 1 CLS + 16 temporal)
```

Nếu `has_face=False` hoặc MediaPipe không detect được mặt → `face_feat = zeros(1, 33, 768)`.

### 2c. Context Encoder — ViT-B/16

**Model:** `google/vit-base-patch16-224` (pretrain ImageNet-21k)

Encode **webcam_crop** (vùng chứa cả thân trên streamer, rộng hơn face_crop) để nắm bắt body language, posture, background. Chỉ fallback về full frame nếu không detect được webcam.

```
webcam_crop (từ YOLO + DBSCAN — Bước 1c)
    │   vùng thân trên streamer, không crop sát mặt
    ▼
    16 frame (crop theo WebcamBBox, resize 224×224)
    │
    ▼ ViT-B/16: CLS token mỗi frame → mean pool
    │
    └── context_feat: (1, 1, 768)
```

Nếu không có webcam → fallback `encode_from_paths()` dùng full frame.

### 2d. Text Encoder — CafeBERT / XLM-R / PhoBERT

**Model mặc định:** `uitnlp/CafeBERT` (cấu hình qua `text_encoder.backend`)

```
transcript: "BOOM! ACE rồi bro!"   ← output từ Bước 1d
    │
    ▼ tokenize → Transformer → CLS token
    │
    └── text_feat: (1, 1, 768)
```

### Tóm tắt output 4 encoder

| Modality | Model | Shape | Ghi chú |
|----------|-------|-------|---------|
| Audio | Whisper encoder-only (whisper-small) | `(1, 64, 768)` | 64 token, ~78ms/token |
| Face | ViT-FER | `(1, 33, 768)` | face_crop từ MediaPipe trong webcam region; zero nếu không có mặt |
| Context | ViT-B/16 | `(1, 1, 768)` | webcam_crop; fallback full frame nếu không có webcam |
| Text | CafeBERT | `(1, 1, 768)` | từ transcript của Whisper |

---

## Bước 3 — Fusion (Conv-Attention 4M)

**Module:** `fusion/conv_attention.py`

```
audio   (1, 64, 768)
face    (1,  1, 768)
context (1,  1, 768)
text    (1,  1, 768)
```

**Align về T=64:** T=1 → broadcast expand; T=64 → giữ nguyên.

Sau align, tạo hai cấu trúc:
```
F_d = concat([audio, face, ctx, text], dim=-1)   → (1, 64, 3072)
F_s = stack( [audio, face, ctx, text], dim=-1)   → (1, 64, 768, 4)
```

**Nhánh Conv** — bắt thay đổi theo thời gian:
```
F_d → Conv1d(3072→768, kernel=3) → ResConvBlock×4 → F_conv (1, 64, 768)
```

**Nhánh Attention** — chọn modality quan trọng tại mỗi timestep:
```
F_d → MLP(3072→1536→4) → Softmax → weights (1, 64, 4)
weights × F_s → F_attn (1, 64, 768)
```

**Kết hợp:**
```
u_fusion = F_conv + F_attn    →    (1, 64, 768)
```

---

## Bước 4 — Classifier

**Module:** `classifiers/mlp.py`

```
u_fusion (1, 64, 768)
    │
    ▼ mean pool → (1, 768)
    │
    ▼ Linear(768→256) + GELU + Dropout(0.3)
    │
    ▼ Linear(256→8)
    │
    └── logits (1, 8)
        argmax → predicted_label
        softmax → confidence (1, 8)
```

Nhãn: `neutral | hype | amused | tilted | sad | shocked | fear | disgusted`

---

## Bước 5 — LLM-1 Reasoning

**Module:** `llm/llm1_explainer.py`  
**Cần:** checkpoint `llm1_explanation_best.pt` (ModalAdapter + g_head từ Stage A training)

LLM-1 chạy theo `_CUE_PROMPT` — LLM nhận **soft tokens** biểu diễn toàn bộ đặc trưng đa phương thức, sau đó tự mô tả các đặc điểm quan sát được và xác định cảm xúc.

### 5a. ModalAdapter — project multimodal embeddings → LLM space

Nhận 5 input, mỗi input có Linear projection riêng → d_llm (3584 cho Qwen2.5-7B):

```python
soft_tokens, attn_mask = modal_adapter(
    fusion_emb,      # (B, T_f, 768) ← u_fusion từ Conv-Attention 4M
    penult=penult,   # (B, 256)      ← output của Linear(768→256)+GELU trong MLP, trước lớp Linear(256→8)
    audio=audio_emb, # (B, 64, 768)  ← Whisper encoder output
    face=face_emb,   # (B, 1, 768)   ← ViT-FER output
    context=ctx_emb, # (B, 1, 768)   ← ViT-B/16 output
    has_face=...,    # (B,) bool     ← mask face nếu không có webcam
)
# Text KHÔNG đưa vào ModalAdapter — LLM xử lý transcript trực tiếp qua prompt text
```

Output là **chuỗi soft tokens** ghép theo thứ tự:

```
soft_tokens: (B, T_total, 3584)

T_total = 1 (penult) + T_f (fusion) + 64 (audio) + 1 (face) + 1 (context)

layout: [penult_tok | fusion_toks | audio_toks | face_toks | ctx_toks]
```

Face tokens bị mask = 0 nếu `has_face=False` hoặc tensor toàn 0.

### 5b. Build prompt với _CUE_PROMPT

```python
_CUE_PROMPT = (
    "Dựa trên đặc trưng đa phương thức, mô tả các đặc điểm quan sát được "
    "và xác định cảm xúc.\n"
    "Trả lời theo format: Cues: face: ...; voice: ...; scene: ...; text: .... "
    "Emotion: [nhãn]."
)
```

Nếu có transcript:
```
prompt = f'Lời nói: "{transcript}"\n{_CUE_PROMPT}'
```

Nếu không có transcript:
```
prompt = _CUE_PROMPT
```

### 5c. Generate

```
soft_tokens (B, T_total, 3584)  ← chuỗi token multimodal (không phải chữ)
    +
text_embeds = embed(prompt)     ← embed văn bản prompt
    │
    ▼ inputs_embeds = cat([soft_tokens, text_embeds], dim=1)
    │
    ▼ Qwen2.5-7B-Instruct.generate(inputs_embeds=inputs_embeds)
    │   LLM không nhận input_ids thông thường
    │   soft_token được attend như token đầu tiên trong chuỗi
    │
    └── raw output:
        "Cues: face: mouth=open, eyes=wide, brows=raised;
               voice: pitch=high, energy=loud, rate=fast;
               scene: high body movement;
               text: exclamation, game_term, 3_words.
         Emotion: hype."
```

### 5d. Parse output

```python
cues_match    = re.search(r"Cues:\s*(.*?)(?:\.\s*Emotion:)", raw, re.DOTALL)
emotion_match = re.search(r"Emotion:\s*(\w+)", raw)

reasoning = cues_match.group(1).strip()    # "face: mouth=open, ...; voice: ..."
answer    = emotion_match.group(1).lower() # "hype"
```

LLM-1 **không override** nhãn từ Classifier — `answer` từ LLM dùng để kiểm tra tính nhất quán, nhãn cuối vẫn là `predicted_label` từ Bước 4.

---

## Output cuối

```json
{
  "clip_id": "streamer1_clip_042",
  "predicted_label": "hype",
  "confidence": 0.87,
  "per_class_scores": {
    "neutral": 0.03, "hype": 0.87, "amused": 0.06,
    "tilted": 0.01, "sad": 0.01, "shocked": 0.01,
    "fear": 0.01,   "disgusted": 0.00
  },
  "llm_reasoning": "face: mouth=open, eyes=wide, brows=raised; voice: pitch=high, energy=loud, rate=fast; text: exclamation, game_term, 3_words",
  "llm_emotion": "hype",
  "format_valid": true
}
```

---

## Luồng đầy đủ theo tensor

```
clip.mp4
    │
    ├── clip.wav ──► Whisper ────────────────────────────────────► transcript (text)
    │       │                                                              │
    │       └────────────────────► Whisper encoder ─────► audio_emb (1,64,768) ─┐
    │                                                                              │
    ├── frames ──► YOLO+DBSCAN ──► webcam_crop ──► MediaPipe ──► face_crop ──► ViT-FER ──► face_emb (1,33,768) ──┤
    │                                     └──────────────────────────────────────► ViT-B/16 ──► ctx_emb  (1, 1,768) ──┤
    │                                                                              │
    └── transcript ──────────────► CafeBERT ────────────► text_emb  (1, 1, 768) ─┘
                                                                     │
                                                           Conv-Attention 4M
                                                           u_fusion (1, 64, 768)
                                                                     │
                                                           MLP Classifier
                                                           predicted_label + confidence
                                                                     │
                                                           ModalAdapter
                                                           soft_token (1, 1, 4096)
                                                                     │
                                              transcript ──► cat([soft_token, text_embeds])
                                                                     │
                                                           Qwen2.5-7B (_CUE_PROMPT)
                                                           "Cues: ... Emotion: hype."
```
