# Pipeline Multimodal Emotion Recognition cho Livestream Game — Phiên Bản Full Upgrade

> **Đề tài:** Phân loại cảm xúc của streamer/reviewer game bằng mô hình multimodal kết hợp **transcript (text) + spectrogram (audio) + biểu cảm khuôn mặt (visual)**, với LLM giải thích kết quả bằng tiếng Việt.
>
> **Phiên bản này:** Tích hợp 6 hướng cải tiến từ 2 paper SOTA mới nhất:
> - **Emotion-LLaMAv2 + MMEVerse** (2026): Conv-Attention pre-fusion, Curriculum training, Multi-agent annotation, Encoder pairing
> - **R1-Omni** (Mar 2025): RLVR với GRPO cho LLM reasoning
>
> **Định hướng:** Đồ án học thuật ambitious, có nhiều đóng góp mới. Open-source, team 2 người, compute Kaggle/Colab + optional RunPod.
>
> **Đóng góp mới chính:**
> 1. **Dual-path visual encoding domain-adapted** cho livestream gaming (face crop + context full-frame) — khắc phục hạn chế của approach full-frame paper Emotion-LLaMAv2 khi áp dụng cho domain ngoài talking-head
> 2. Áp dụng Conv-Attention pre-fusion (mới SOTA 2026) cho 4 modality (audio/face/context/text)
> 3. So sánh có hệ thống 4 setup LLM (Explainer / Co-Reasoner / VLM end-to-end / **RLVR-trained**)
> 4. Multi-agent annotation pipeline 100% open-source cho tiếng Việt
> 5. Curriculum learning Perception→Cognition cho VN emotion domain
> 6. Benchmark dataset livestream game VN đầu tiên với 4 modality + reasoning annotation

---

## Mục Lục

1. [Tổng Quan Pipeline (Full Upgrade)](#1-tổng-quan-pipeline-full-upgrade)
2. [Stage 0 — Data Preparation với Multi-Agent Annotation](#2-stage-0--data-preparation-với-multi-agent-annotation)
3. [Stage 1 — Tiền Xử Lý](#3-stage-1--tiền-xử-lý)
4. [Stage 2a — Audio → Spectrogram → Encoder](#4-stage-2a--audio--spectrogram--encoder)
5. [Stage 2b — Video → Face → Encoder](#5-stage-2b--video--face--encoder)
6. [Stage 2c — ASR → Text → Encoder](#6-stage-2c--asr--text--encoder)
7. [Stage 3 — Conv-Attention Pre-Fusion](#7-stage-3--conv-attention-pre-fusion-mới)
8. [Stage 4 — Emotion Classifier](#8-stage-4--emotion-classifier)
9. [Stage 5 — LLM Reasoner (4 Setup, Có RLVR)](#9-stage-5--llm-reasoner-4-setup-có-rlvr)
10. [Perception-to-Cognition Curriculum Training](#10-perception-to-cognition-curriculum-training-mới)
11. [Chi Tiết Engineering Thực Tế](#11-chi-tiết-engineering-thực-tế)
12. [Thiết Kế Thử Nghiệm Tổng Thể](#12-thiết-kế-thử-nghiệm-tổng-thể)
13. [Timeline 14 Tuần Chi Tiết](#13-timeline-14-tuần-chi-tiết)
14. [Phương Án Compute (Có vs Không Có A100)](#14-phương-án-compute-có-vs-không-có-a100)
15. [Checklist Trước Khi Bắt Đầu](#15-checklist-trước-khi-bắt-đầu)

---

## 1. Tổng Quan Pipeline (Full Upgrade)

### 1.1. Sơ đồ kiến trúc mới

```
                    ┌──────────────────────────────────────────┐
                    │  Stage 0 · Data prep                     │
                    │  Crawl + Multi-agent annotation pipeline │
                    │  (Qwen-VL + Qwen-Audio + Qwen2.5-72B)   │
                    └──────────────────────┬───────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────┐
                    │  Stage 1 · Tiền xử lý                    │
                    │  Tách 3 stream song song                 │
                    └──────────────────────┬───────────────────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            │                              │                              │
   ┌────────▼────────┐         ┌───────────▼─────────┐         ┌──────────▼─────────┐
   │  Stage 2a       │         │  Stage 2b           │         │  Stage 2c          │
   │  Audio          │         │  Visual DUAL-PATH   │         │  Text              │
   │  → spectrogram  │         │  (DOMAIN-ADAPTED)   │         │  → Whisper/        │
   │  → AST encoder  │         │ ┌─────────────────┐ │         │    PhoWhisper ASR  │
   │                 │         │ │ Path 1: FACE    │ │         │  → BARTpho (opt)   │
   │                 │         │ │ Webcam detect   │ │         │  → XLM-R/PhoBERT   │
   │                 │         │ │ + crop+ViT-FER  │ │         │    encoder         │
   │                 │         │ │ → h_face        │ │         │  (config-switched) │
   │                 │         │ ├─────────────────┤ │         │                    │
   │                 │         │ │ Path 2: CONTEXT │ │         │                    │
   │                 │         │ │ Full-frame ViT  │ │         │                    │
   │                 │         │ │ (gameplay info) │ │         │                    │
   │                 │         │ │ → h_ctx         │ │         │                    │
   │  → h_audio      │         │ └─────────────────┘ │         │  → h_text          │
   └────────┬────────┘         └──────────┬──────────┘         └─────────┬──────────┘
            │                             │                              │
            └─────────────────────────────┼──────────────────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │ Stage 3 · Conv-Attention Pre-Fusion  │
                       │ (THAY MULT — từ Emotion-LLaMAv2)     │
                       │ 4 MODALITY: audio/face/ctx/text      │
                       │  ┌──────────┐    ┌──────────┐        │
                       │  │ Conv     │    │ Attention│        │
                       │  │ branch   │  + │ branch   │        │
                       │  │ (local)  │    │ (global) │        │
                       │  └──────────┘    └──────────┘        │
                       │            → u_fusion                 │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────┴───────────────────┐
                       │                                      │
              ┌────────▼─────────┐                ┌───────────▼──────────────┐
              │  Stage 4         │                │  Stage 5 · LLM Reasoner  │
              │  Classifier      │                │  4 setup so sánh:        │
              │  MLP → nhãn      │                │  • LLM-1: Explainer       │
              │                  │                │  • LLM-2: Co-Reasoner     │
              │                  │                │  • LLM-3: VLM end-to-end  │
              │                  │                │  • LLM-4: RLVR-trained    │
              └──────────────────┘                └───────────────────────────┘

         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         CURRICULUM TRAINING (xuyên suốt Stage 4-5):
         Stage 1 — Perception:  chỉ train emotion recognition
         Stage 2 — Cognition:   joint train recognition + reasoning
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 1.2. Thay đổi quan trọng so với pipeline cũ

| Khía cạnh | Pipeline cũ | Pipeline Full Upgrade |
|---|---|---|
| **Fusion module** | Cross-modal Transformer (MULT) | **Conv-Attention** (Conv + Attention branch) |
| **Visual encoding** | MediaPipe + crop + ViT (1 path) | **Dual-path: Face crop + Context full-frame** |
| **Domain adaptation** | Generic | **Livestream gaming-specific** |
| **Training** | Single-stage joint | **Curriculum 2-stage** (Perception → Cognition) |
| **LLM setups** | 3 (Explainer/Co-Reasoner/VLM) | **4** (thêm RLVR-trained) |
| **Annotation** | 2 người manual | **Multi-agent pipeline + human verify** |
| **Reasoning data** | Không có (hoặc rất ít) | **Auto-generated reasoning** từ multi-agent |
| **Encoder pairing** | Cố định | **Ablation 5 combinations** |
| **Ablation depth** | 8 experiments | **18+ experiments** |

### 1.3. Bảng tóm tắt

| Stage | Input | Output | Model chính | Open-source? |
|---|---|---|---|---|
| **0. Data prep** | URL video | Clip + label + reasoning | Multi-agent (Qwen-VL/Audio/72B) | ✅ |
| **1. Demuxing** | MP4 | audio.wav + frames | ffmpeg | ✅ |
| **2a. Audio→Spec→Enc** | wav 16kHz | h_audio (768d) | librosa + **AST** | ✅ |
| **2b. Visual Dual-Path** | frames | h_face + h_ctx (2×768d) | **Webcam detect + ViT-FER + ViT-ImageNet** | ✅ |
| **2c. ASR→Text→Enc** | wav | h_text (768d) | **Whisper/PhoWhisper + BARTpho(opt) + XLM-R/PhoBERT** | ✅ |
| **3. Pre-Fusion** | 4 × 768d | u_fusion (768d) | **Conv-Attention module** | ✅ |
| **4. Classifier** | u_fusion | label probs | MLP 2-layer | ✅ |
| **5. LLM** | label + features | Giải thích VN | **Qwen2.5-7B** (4 setup) | ✅ |
| **Training** | Multi-stage | Curriculum | **Perception → Cognition** | ✅ |

---

## 2. Stage 0 — Data Preparation với Multi-Agent Annotation

### 2.1. Input/Output

| | Mô tả |
|---|---|
| **Input** | Danh sách URL kênh YouTube/Facebook Gaming của streamer VN |
| **Output** | 500-800 clip MP4 (5 giây) + CSV nhãn + **reasoning annotations** |
| **Công cụ** | `yt-dlp`, `ffmpeg`, **Label Studio**, **Multi-agent pipeline** |

### 2.2. Phân chia game genre

Giữ nguyên như pipeline cũ:
- MOBA (LMHT, Lien Quan): 25%
- FPS (Valorant, CSGO, PUBG): 25%
- Horror: 15%
- Casual/Party: 15%
- RPG/Adventure: 10%
- Mobile games: 10%

### 2.3. Multi-Agent Annotation Pipeline (CẢI TIẾN MỚI)

**Lấy ý tưởng từ Emotion-LLaMAv2 (2026):** thay vì chỉ annotate emotion label, tự động sinh thêm **multimodal reasoning description** dùng nhiều LLM/VLM open-source.

#### Pipeline 6 bước

```
Step 1: Peak Frame Detection
  Input: video clip
  Output: frame với highest Action Unit intensity
  Tool: OpenFace 2.0 (extract AU intensities)

Step 2: Visual Expression Description (Cved)
  Input: peak frame
  Output: mô tả facial AU
  Tool: OpenFace → AU list (vd: "AU12 intensity 4.2, AU6 intensity 3.8")

Step 3: Visual Objective Description (Cvod)
  Input: peak frame
  Output: mô tả scene/context
  Tool: Qwen2.5-VL-72B-Instruct (quantize 4-bit) hoặc Qwen2.5-VL-32B
  Prompt: "Mô tả cảnh và bối cảnh trong frame này bằng tiếng Việt, tập trung vào
          tư thế, hành động, môi trường xung quanh streamer."

Step 4: Audio Tone Description (Catd)
  Input: audio.wav
  Output: mô tả prosody (pitch, energy, speaking rate)
  Tool chính: Qwen2-Audio-7B-Instruct
  Tool backup: Audio-Reasoner-7B
  Prompt: "Phân tích đặc điểm âm thanh: tốc độ nói, cao độ giọng,
          năng lượng, có tiếng la/cười không."

Step 5: Linguistic Subtitles (Cls)
  Input: audio.wav
  Output: transcript
  Tool: Whisper-large-v3 (đã có ở Stage 2c)

Step 6: Multimodal Consolidation (Cmd) — KEY STEP
  Input: Cved + Cvod + Catd + Cls + emotion label
  Output: coherent reasoning description bằng tiếng Việt
  Tool: Qwen2.5-72B-Instruct (quantize 4-bit, ~40GB VRAM hoặc 32B q4 ~20GB)
  Backup: Qwen2.5-32B-Instruct nếu compute hạn chế
```

**Tại sao dùng Qwen2.5-72B-Instruct (không phải GPT-4o):**
- Mở source 100%, không tốn budget API
- Quantize 4-bit còn ~40GB VRAM → cần 1 GPU 48GB (A6000) hoặc 2 GPU 24GB
- Chất lượng VN khá tốt, kém GPT-4o ~10% nhưng đủ cho annotation
- **Fallback:** Qwen2.5-32B-Instruct q4 chỉ cần ~20GB → 1 RTX 4090 hoặc T4 x2

### 2.4. Implementation chi tiết

**Khởi tạo multi-agent pipeline:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Agent 1: Qwen2.5-VL cho visual objective
from transformers import Qwen2_5_VLForConditionalGeneration
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",  # 32B nếu compute hạn chế
    quantization_config=bnb_4bit,
    device_map="auto"
)

# Agent 2: Qwen2-Audio cho prosody
audio_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    quantization_config=bnb_4bit,
    device_map="auto"
)

# Agent 3: Qwen2.5-72B làm consolidator
consolidator = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",  # 32B q4 ~20GB
    quantization_config=bnb_4bit,
    device_map="auto"
)
```

**Pipeline annotation full:**

```python
def annotate_clip(clip_path, emotion_label):
    """
    Annotate 1 clip với multi-agent pipeline.
    Output: dict gồm label + reasoning description.
    """
    # Step 1: Peak frame detection
    peak_frame, au_intensities = detect_peak_frame_openface(clip_path)
    
    # Step 2: Visual expression (Cved)
    Cved = format_au_description(au_intensities)
    # Vd: "AU12 (lip corner puller) intensity 4.2, AU6 (cheek raiser) 3.8"
    
    # Step 3: Visual objective (Cvod)
    prompt_vod = "Mô tả ngắn gọn cảnh trong frame này bằng tiếng Việt..."
    Cvod = vl_model.generate(peak_frame, prompt_vod)
    
    # Step 4: Audio tone (Catd)
    prompt_atd = "Phân tích âm thanh: pitch, energy, speaking rate..."
    Catd = audio_model.generate(clip_path, prompt_atd)
    
    # Step 5: Lexical subtitles (Cls)
    Cls = whisper_transcribe(clip_path)
    
    # Step 6: Consolidation
    consolidation_prompt = f"""
Bạn là chuyên gia phân tích cảm xúc. Hãy viết một đoạn reasoning 
ngắn (3-5 câu) bằng tiếng Việt, giải thích vì sao streamer này đang 
trong trạng thái {emotion_label}.

Bằng chứng đa phương thức:
- Khuôn mặt (Action Units): {Cved}
- Cảnh: {Cvod}
- Giọng nói: {Catd}
- Lời nói: "{Cls}"

Format output:
<think>
[Phân tích từng modality và liên kết chúng]
</think>
<answer>{emotion_label}</answer>
"""
    reasoning = consolidator.generate(consolidation_prompt)
    
    return {
        "clip_id": Path(clip_path).stem,
        "emotion_label": emotion_label,
        "Cved": Cved,
        "Cvod": Cvod,
        "Catd": Catd,
        "Cls": Cls,
        "reasoning": reasoning
    }
```

### 2.5. Human Verification

Sau khi multi-agent sinh xong, **2 thành viên team verify** thay vì write từ đầu:

```
Workflow:
1. Multi-agent sinh ~500 clip × reasoning → ~2-3 ngày compute
2. 2 thành viên review:
   - 100% verify emotion label (đảm bảo κ ≥ 0.6)
   - 30% verify reasoning quality (sample random)
   - Flag bad reasoning để regenerate
3. Final dataset: 500 clip với label + reasoning
```

**Lợi ích vs annotation manual hoàn toàn:**
- Giảm ~70% effort viết reasoning
- Consistency cao hơn (LLM dùng cùng prompt)
- Có **reasoning supervision** cho LLM-2 và LLM-3 training
- Scale dễ → có thể lên 1000+ clip nếu cần

### 2.6. Schema dán nhãn

**Schema A — Ekman 7 (tham khảo):** `vui, buồn, tức giận, sợ hãi, ngạc nhiên, ghê tởm, trung tính`

**Schema B — Gaming-specific 9 (gaming_9) — schema chính thức (`n_classes=9`):**

| # | Nhãn | Tiếng Việt | Đặc điểm |
|---|------|-----------|----------|
| 0 | neutral   | Trung tính   | Idle, giải thích, chờ |
| 1 | focus     | Tập trung    | Tryhard, căng thẳng, nghiêm túc |
| 2 | hype      | Phấn khích   | Clutch, ace, thắng lớn |
| 3 | amused    | Hài hước     | Cười, khoảnh khắc buồn cười |
| 4 | tilted    | Cay cú       | Tức giận, thất vọng, toxic |
| 5 | sad       | Buồn         | Thua, hối hận, chán nản |
| 6 | shocked   | Sốc          | Ngạc nhiên mạnh, jump scare |
| 7 | fear      | Sợ hãi       | Game kinh dị, hoảng loạn |
| 8 | disgusted | Ghê tởm      | Gore, đồng đội tệ, cringe |

> **Lưu ý:** Toàn bộ codebase dùng `gaming_9` với `n_classes=9`. Schema Ekman 7 chỉ dùng trong ablation so sánh.

### 2.7. Thử nghiệm Stage 0

| Ablation | Mục đích |
|---|---|
| **Multi-agent vs Manual annotation** | Đo gap chất lượng (subset 50 clip làm cả 2 cách) |
| **Qwen2.5-72B vs Qwen2.5-32B consolidator** | Có cần model lớn nhất không? |
| **With/without OpenFace AU** | AU description có thiết yếu không? |
| **Ekman vs Gaming-specific** | Schema nào dễ annotate hơn (đo κ) |

---

## 3. Stage 1 — Tiền Xử Lý

Giữ nguyên như pipeline cũ — không có cải tiến từ paper.

### 3.1. Input/Output

| | Mô tả |
|---|---|
| **Input** | 1 clip MP4 |
| **Output** | `audio.wav` 16kHz mono + `frames/*.jpg` @ 4fps |
| **Công cụ** | `ffmpeg`, OpenCV |

### 3.2. Code mẫu

```python
import subprocess, cv2
from pathlib import Path

def preprocess_clip(clip_path, output_dir):
    clip_id = Path(clip_path).stem
    out = Path(output_dir) / clip_id
    out.mkdir(parents=True, exist_ok=True)
    
    # Tách audio 16kHz mono
    subprocess.run([
        "ffmpeg", "-i", clip_path,
        "-ar", "16000", "-ac", "1",
        "-y", str(out / "audio.wav")
    ], check=True)
    
    # Tách frames @ 4fps
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps / 4)
    saved = 0
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            cv2.imwrite(str(out / f"frame_{saved:04d}.jpg"), frame)
            saved += 1
    cap.release()
    return out
```

### 3.3. Tách nhạc nền (optional với Demucs)

Chỉ khi audio EDA cho thấy nhạc nền quá lớn. Đa số clip không cần.

---

## 4. Stage 2a — Audio → Spectrogram → Encoder

### 4.1. Bước 2a.1: Log-Mel Spectrogram

Giữ nguyên như pipeline cũ.

| | Mô tả |
|---|---|
| **Input** | `audio.wav` (16kHz, ~30s) |
| **Output** | Tensor `[3, 224, 224]` (mel spectrogram → fake 3-channel image) |
| **Tham số** | win=25ms, hop=10ms, n_mels=128, log scale + z-score |

### 4.2. Bước 2a.2: Encoder

| | Mô tả |
|---|---|
| **Input** | Tensor `[3, 224, 224]` |
| **Output** | `h_audio` ∈ ℝ⁷⁶⁸ |
| **Model chính** | **AST** — `MIT/ast-finetuned-audioset-10-10-0.4593` |

**Implementation:**
```python
from transformers import ASTFeatureExtractor, ASTModel

feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
ast_model = ASTModel.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

def encode_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = ast_model(**inputs)
    # Adaptive pooling về 64 tokens (Emotion-LLaMAv2 finding)
    h_audio_seq = outputs.last_hidden_state  # [1, N, 768]
    h_audio_seq = adaptive_pool_audio(h_audio_seq, target_len=64)  # [1, 64, 768]
    return h_audio_seq
```

**Tại sao 64 tokens:** Emotion-LLaMAv2 ablation cho thấy 64 audio tokens là sweet spot (Figure 3a của paper).

### 4.3. Thử nghiệm Stage 2a

| Ablation | Setup | Mục đích |
|---|---|---|
| **Audio token count** | 1/2/4/8/16/32/64/128 tokens | Tìm sweet spot cho domain game (kỳ vọng ~64) |
| **AST vs Wav2Vec2** | Spectrogram vs waveform | Hướng tiếp cận nào tốt hơn |
| **AST vs HuBERT** | Different audio backbone | Replicate finding của paper |
| **n_mels: 64/128/256** | Resolution mel | Balance |

---

## 5. Stage 2b — Video → Face + Context → Encoder (DOMAIN-ADAPTED)

### 5.1. Tại Sao Không Follow Paper Máy Móc

**Quan sát quan trọng:** Emotion-LLaMAv2 bỏ face crop và dùng full-frame, **nhưng cách này KHÔNG phù hợp cho domain livestream game của chúng ta**. Đây là một điểm cần phân tích kỹ vì nó là **decision point critical** ảnh hưởng accuracy.

#### Tại sao paper Emotion-LLaMAv2 bỏ face crop work?

Paper test trên MMEVerse (12 dataset aggregate): IEMOCAP, MELD, DFEW, MAFW, CMU-MOSEI... Tất cả đều là **talking-head video** — khuôn mặt chiếm 60-85% frame:

| Dataset của paper | Loại video | % face trong frame |
|---|---|---|
| IEMOCAP | Lab studio, diễn viên | ~80% close-up |
| MELD | Friends TV series | ~60% medium shot |
| DFEW/MAFW | Movies | ~70% close-up + medium |
| CMU-MOSEI | YouTube monologue | ~85% talking head |

Trong các domain này, full-frame ≈ face crop về mặt content. **Bỏ face detection** chỉ tránh error propagation, không mất signal.

#### Tại sao domain livestream game KHÁC HOÀN TOÀN?

```
Frame typical livestream game:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│                                                          │
│            GAME SCREEN (~85-90% frame)                  │
│            - Gameplay, HUD, score                        │
│            - Có thể có face nhân vật game               │
│            - Cutscene close-up trong RPG/Horror         │
│                                                          │
│                                          ┌────────┐      │
│                                          │WEBCAM  │      │
│                                          │STREAMER│      │
│                                          │ ~10-15%│      │
│                                          └────────┘      │
└──────────────────────────────────────────────────────────┘
```

**Streamer webcam chỉ chiếm 10-15% frame.** 4 vấn đề nghiêm trọng nếu áp dụng full-frame:

**(1) Diluted signal-to-noise ratio**
- 85-90% pixel là gameplay (noise cho emotion task)
- 10-15% pixel là khuôn mặt streamer (signal thực)
- Tỷ lệ signal/noise ≈ 1/8 (so với 3/1 đến 5/1 trong data của paper)
- ViT-FER pretrain AffectNet (toàn ảnh khuôn mặt cận cảnh) không có inductive bias để ignore 85% noise

**(2) Confusion với face nhân vật game (CRITICAL)**
Nhiều genre có face nhân vật chiếm phần lớn frame:

| Genre | Risk confusion |
|---|---|
| MOBA (LMHT, LQ) | Trung bình — top-down view |
| FPS (Valorant, CSGO) | Trung bình — first-person, character nhỏ |
| **Horror (RE, Outlast)** | **CAO** — jumpscare NPC close-up |
| **RPG (Genshin, Persona)** | **CAO** — cutscene close-up nhân vật |
| Casual (Stumble Guys) | Thấp — cartoon style |

Kịch bản problem:
```
Cutscene game: face NPC gào thét (40% frame)
+ Webcam: streamer đang cười thích thú (10%)
→ Ground truth: Amused (streamer thấy buồn cười)
→ Full-frame ViT-FER có thể predict: Angry/Fear (do face NPC dominate)
```

**(3) Variable webcam location**
Streamer đặt webcam bất kỳ vị trí nào: góc dưới phải/trái, góc trên, facecam mode (full-screen face). Cross-modal attention phải học position-invariant face attention → khó với 500-800 clip.

**(4) Distribution shift**
- ViT-FER pretrain trên 400k ảnh AffectNet, toàn frontal face cận cảnh
- Full-frame livestream có distribution hoàn toàn khác → pretrained weights không tận dụng được đầy đủ

#### Kết luận

**Không follow paper máy móc. Domain matters.** Cần adapt approach cho livestream game.

### 5.2. Approach Đề Xuất: Dual-Path Visual Encoding

Thay vì full-frame như paper, dùng **2 path song song**:

```
                    ┌──────────────────────┐
                    │  Full frame raw      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌──────────────────┐              ┌──────────────────┐
    │ Path 1: FACE     │              │ Path 2: CONTEXT  │
    │ (chính)          │              │ (phụ)            │
    │                  │              │                  │
    │ Webcam detection │              │ Full-frame       │
    │ → Crop streamer  │              │ → Resize 224x224 │
    │ → ViT-FER        │              │ → General ViT    │
    │                  │              │                  │
    │ → h_face (768d)  │              │ → h_ctx (768d)   │
    └──────────────────┘              └──────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                Đưa vào Conv-Attention fusion
                (4 modality: face/ctx/audio/text)
```

**Vai trò 2 path:**
- **Path 1 — Face:** carry chính cho emotion. Clean signal sau khi crop streamer face. Distribution match với ViT-FER pretrain.
- **Path 2 — Context:** carry thông tin gameplay bổ sung. Vd: "đang fail boss" (frustrated context) vs "đang win" (hype context). Giúp model phân biệt cảm xúc tương tự nhưng context khác (vd: scream khi sợ vs scream khi phấn khích).

**Đóng góp đặc thù:** Đây là **adaptation domain-specific** chưa thấy trong literature multimodal emotion recognition cho livestream gaming. Có thể publish như methodological contribution.

### 5.3. Path 1 — Webcam Detection + Face Encoding

#### Bước 1: Webcam Region Detection

Detect vị trí webcam streamer trong frame. Webcam có 2 đặc điểm phân biệt với face nhân vật game:
- **Vị trí ổn định** xuyên suốt clip (streamer không di chuyển webcam)
- **Xuất hiện liên tục**, không sporadic như face NPC

**Implementation:**

```python
import mediapipe as mp
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class WebcamDetector:
    """
    Detect webcam region trong livestream frame.
    Phân biệt webcam streamer với face nhân vật game qua spatial stability.
    """
    def __init__(self):
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7
        )
    
    def detect_webcam_region(self, clip_path, sample_n=30):
        """
        Trả về bbox (x, y, w, h) của webcam (normalized) hoặc None.
        """
        cap = cv2.VideoCapture(clip_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample đều
        indices = np.linspace(0, total_frames - 1, sample_n).astype(int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret: frames.append(frame)
        cap.release()
        
        if not frames:
            return None
        
        # Detect face trong tất cả sample frames
        detections = []  # list of (xmin, ymin, w, h, frame_idx)
        for idx, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.mp_face.process(rgb)
            if result.detections:
                for det in result.detections:
                    bbox = det.location_data.relative_bounding_box
                    detections.append((
                        bbox.xmin, bbox.ymin, 
                        bbox.width, bbox.height,
                        idx
                    ))
        
        if not detections:
            # Streamer không hiện mặt (no facecam)
            return None
        
        # Cluster detections theo position (xmin, ymin)
        # Webcam → bbox xuất hiện ở vị trí ổn định (cluster lớn)
        # Game NPC face → sporadic (noise hoặc cluster nhỏ)
        coords = np.array([[d[0], d[1]] for d in detections])
        clustering = DBSCAN(eps=0.05, min_samples=5).fit(coords)
        
        labels = clustering.labels_
        if all(l == -1 for l in labels):
            # Toàn noise, không có face stable → có thể không có webcam
            # Fallback: lấy detection xuất hiện nhiều nhất
            return None
        
        # Lấy cluster lớn nhất (loại noise -1)
        valid_labels = [l for l in set(labels) if l != -1]
        largest_cluster = max(valid_labels, key=lambda c: list(labels).count(c))
        
        # Tính stability score
        cluster_size = list(labels).count(largest_cluster)
        stability_ratio = cluster_size / sample_n
        
        if stability_ratio < 0.5:
            # Cluster không đủ stable → có thể là NPC face
            return None
        
        # Average bbox của cluster lớn nhất
        cluster_dets = [detections[i] for i, l in enumerate(labels) 
                        if l == largest_cluster]
        avg_xmin = np.mean([d[0] for d in cluster_dets])
        avg_ymin = np.mean([d[1] for d in cluster_dets])
        avg_w = np.mean([d[2] for d in cluster_dets])
        avg_h = np.mean([d[3] for d in cluster_dets])
        
        return (avg_xmin, avg_ymin, avg_w, avg_h)
```

#### Bước 2: Face Crop từ Webcam Region

```python
def extract_streamer_face(frame, webcam_bbox, target_size=(224, 224), 
                          margin=0.2):
    """
    Crop khuôn mặt streamer từ webcam region.
    """
    if webcam_bbox is None:
        # Fallback: dùng full-frame downsample (last resort)
        return cv2.resize(frame, target_size)
    
    h, w = frame.shape[:2]
    xmin, ymin, bw, bh = webcam_bbox
    
    # Convert normalized → pixel
    x = int(xmin * w)
    y = int(ymin * h)
    bw_px = int(bw * w)
    bh_px = int(bh * h)
    
    # Expand margin để bắt cả vùng quanh mặt (tai, cổ, bối cảnh gần)
    x = max(0, x - int(bw_px * margin))
    y = max(0, y - int(bh_px * margin))
    bw_px = min(w - x, int(bw_px * (1 + 2 * margin)))
    bh_px = min(h - y, int(bh_px * (1 + 2 * margin)))
    
    face_crop = frame[y:y+bh_px, x:x+bw_px]
    return cv2.resize(face_crop, target_size)
```

#### Bước 3: Face Encoding với Dual-View

Vẫn giữ ý tưởng dual-view (global + temporal) từ Emotion-LLaMAv2, **nhưng áp dụng trên face crop**, không full-frame:

```python
from transformers import ViTModel, AutoImageProcessor

class StreamerFaceEncoder(nn.Module):
    """
    Path 1: Face encoder với dual-view (global + temporal).
    Input: face crops đã extract từ webcam region.
    """
    def __init__(self):
        super().__init__()
        # Global view: middle frame face
        self.global_encoder = ViTModel.from_pretrained(
            "trpakov/vit-face-expression"  # ViT-FER AffectNet
        )
        # Temporal view: 16 frame faces  
        # Có thể dùng cùng model hoặc TimeSformer
        self.temp_encoder = ViTModel.from_pretrained(
            "trpakov/vit-face-expression"
        )
    
    def forward(self, face_crops):
        """
        face_crops: list of N face crops (đã extract qua WebcamDetector)
        """
        # Global view: middle frame
        middle = face_crops[len(face_crops) // 2]
        h_glo = self.global_encoder(middle).last_hidden_state[:, 0]
        
        # Temporal view: 16 frame uniformly sampled
        if len(face_crops) > 16:
            indices = np.linspace(0, len(face_crops) - 1, 16).astype(int)
            sampled = [face_crops[i] for i in indices]
        else:
            sampled = face_crops
        
        temp_outputs = self.temp_encoder(torch.stack(sampled))
        # Spatial pool 2x2 + temporal pool
        h_temp_seq = spatial_pool_2x2(temp_outputs.last_hidden_state)
        h_temp = h_temp_seq.mean(dim=1)  # temporal pool
        
        return h_glo, h_temp
```

### 5.4. Path 2 — Context Encoding (Gameplay Information)

Path này giữ full-frame để capture gameplay context:

```python
class GameContextEncoder(nn.Module):
    """
    Path 2: Context encoder cho gameplay scene.
    Input: full frame (downsampled).
    """
    def __init__(self):
        super().__init__()
        # Dùng general ViT, không phải ViT-FER
        self.encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"  # ImageNet pretrain
        )
    
    def forward(self, full_frames):
        """
        full_frames: 16 frame uniformly sampled, đã resize 224x224
        """
        outputs = self.encoder(full_frames)
        # CLS token làm context representation
        h_ctx_seq = outputs.last_hidden_state[:, 0]  # [16, 768]
        h_ctx = h_ctx_seq.mean(dim=0, keepdim=True)  # [1, 768]
        return h_ctx
```

**Tại sao dùng ViT pretrain ImageNet (không phải ViT-FER) cho Path 2:**
- Context encoder không cần focus face → ViT-FER waste capacity
- ImageNet ViT có inductive bias cho scene/object → phù hợp gameplay
- Pretrain rộng hơn → robust với HUD, UI, character model

### 5.5. Edge Cases Handling

#### Case 1: Streamer "no facecam" mode

Một số streamer (đặc biệt game competitive) **không bật webcam**. WebcamDetector trả về `None`.

**Strategy:**
```python
def encode_visual(clip_path):
    webcam_bbox = webcam_detector.detect_webcam_region(clip_path)
    
    if webcam_bbox is None:
        # No facecam mode: chỉ dùng Path 2 (context)
        # Audio + text vẫn carry cảm xúc
        h_face_glo = torch.zeros(1, 768)  # zero placeholder
        h_face_temp = torch.zeros(1, 768)
        h_ctx = context_encoder(...)
        # Flag để model biết face modality missing
        return h_face_glo, h_face_temp, h_ctx, has_face=False
    else:
        # Normal mode: cả 2 path
        ...
        return h_face_glo, h_face_temp, h_ctx, has_face=True
```

→ Combine với **modality dropout** trong fusion: model học robust khi face missing.

#### Case 2: Facecam mode (full-screen face)

Một số streamer review game (vd: PewPew reaction) → face chiếm 50%+ frame.

**Strategy:** WebcamDetector tự handle vì cluster face sẽ rất lớn. Bbox extract ra ≈ full-frame.

#### Case 3: Webcam di chuyển trong clip

Hiếm, nhưng có (vd: thay đổi layout giữa stream). Hiện tại detector tính average bbox → có thể sai.

**Strategy:** Cho phép multiple clusters và dùng nearest cluster cho mỗi frame:
```python
# Thay vì 1 bbox cho toàn clip, dùng per-frame bbox lookup
def detect_webcam_per_frame(self, frames):
    ...
    return {frame_idx: bbox_for_that_frame}
```

#### Case 4: Confusion với face nhân vật trong cutscene

Khi cutscene chiếm phần lớn clip → face NPC stable → có thể bị detect nhầm thành webcam.

**Strategy:** 
- Webcam thường có **vị trí ở rìa frame** (góc dưới phải/trái, etc.)
- NPC face thường ở **giữa frame**
- Bias detection ưu tiên bbox ở edge regions

```python
def is_likely_webcam(bbox):
    xmin, ymin, w, h = bbox
    xcenter, ycenter = xmin + w/2, ymin + h/2
    # Webcam thường ở góc (center cách edge < 0.3)
    edge_dist = min(xcenter, ycenter, 1-xcenter, 1-ycenter)
    return edge_dist < 0.3
```

### 5.6. Ablation Critical — So Sánh 3 Strategies

**Đây là phần đóng góp methodological quan trọng nhất của Stage 2b.** So sánh 3 approach để chứng minh choice của ta đúng:

| Strategy | Mô tả | Expected (overall) | Expected (per-genre) |
|---|---|---|---|
| **A. Full-frame (Emotion-LLaMAv2)** | Resize full-frame → ViT-FER | Baseline | Suffer trên Horror/RPG |
| **B. Face crop only** | Webcam detect + crop face | Tốt hơn A | Consistent across genre |
| **C. Dual-path (face + context)** | Path 1 + Path 2 song song | **Tốt nhất** | Robust nhất |

**Per-genre breakdown ablation:**

| Genre | A (Full-frame) | B (Face only) | C (Dual-path) | Expected gap |
|---|---|---|---|---|
| MOBA | Low risk | Tốt | Tốt | Small (~1-2%) |
| FPS | Low risk | Tốt | Tốt | Small |
| **Horror** | **Suffer** | Tốt | Tốt | **Lớn (5-10%)** |
| **RPG** | **Suffer** | Tốt | Tốt | **Lớn** |
| Casual | Low risk | Tốt | Tốt | Small |

→ Bảng này sẽ là **figure key trong báo cáo** — chứng minh follow paper máy móc không phải lúc nào cũng đúng cho domain mới.

### 5.7. Implementation Strategy theo Timeline

**Tuần 5-6 (Baseline):**
- Implement Strategy B (Face crop only)
- Validate WebcamDetector trên 50 clip pilot
- Đo failure rate của face detection

**Tuần 9-10 (Nâng cấp):**
- Implement Strategy C (Dual-path)
- Train với Conv-Attention fusion 4 modality (face/ctx/audio/text)

**Tuần 11 (Ablation key):**
- Train cả 3 strategies (A, B, C) với cùng setup
- Per-genre breakdown
- Visualize attention maps để show model focus đâu

### 5.8. Encoder Pairing Experiments (vẫn giữ từ pipeline cũ)

Sau khi chọn được Strategy (kỳ vọng C), ablate encoder combinations:

| Audio | Face encoder (Path 1) | Context encoder (Path 2) | Mục đích |
|---|---|---|---|
| AST | ViT-FER | ViT-ImageNet | Default proposed |
| AST | ViT-FER | EVA | Strong context encoder |
| AST | EVA | ViT-ImageNet | Strong face encoder |
| Wav2Vec2-VN | ViT-FER | ViT-ImageNet | VN-specific audio |
| HuBERT | EVA | EVA | Paper recommendation (adapted) |

### 5.9. Thử nghiệm Stage 2b Tổng Hợp

| Ablation | Setup | Mục đích |
|---|---|---|
| **Strategy comparison (KEY)** | A/B/C trên cùng split | Chứng minh dual-path tốt nhất cho domain |
| **Per-genre breakdown (KEY)** | A/B/C × {MOBA, FPS, Horror, RPG, Casual} | Show genre-specific issues của A |
| **Webcam detection success rate** | Đo trên 600 clip | Validate detector reliability |
| **No-facecam handling** | Subset clip không có webcam | Test fallback strategy |
| **Encoder pairing** | 5 combinations | Best synergy |
| **Number of frames** | 1/2/4/8/16/32 | Sweet spot cho domain game |
| **Margin in face crop** | 0.1 / 0.2 / 0.3 | Bao nhiêu context quanh face |

### 5.10. Tóm Tắt Stage 2b

| Aspect | Pipeline cũ (full-frame paper) | Pipeline updated (dual-path) |
|---|---|---|
| **Number of paths** | 1 (full-frame) | **2 (face + context)** |
| **Face handling** | Implicit attention | **Explicit webcam detect + crop** |
| **Context handling** | Trộn với face | **Tách riêng path 2** |
| **Domain adaptation** | Không | **Có (livestream-specific)** |
| **Robustness no-facecam** | OK | **OK với fallback** |
| **Resist nhân vật game confusion** | **Không** | **Có** |
| **Đóng góp paper** | Replicate | **Methodological adaptation** |

---

## 6. Stage 2c — ASR → Text → Encoder

Stage 2c được nâng cấp đáng kể so với pipeline cũ: hỗ trợ 2 ASR backend (Whisper / PhoWhisper), optional BARTpho post-processing, và 2 text encoder có thể đổi qua config (`xlmr` | `phobert`). Toàn bộ lựa chọn được điều khiển qua `config.yaml` mà không cần sửa code.

### 6.1. ASR Backend — Chọn qua config

**Vấn đề với Whisper thuần:**  
Whisper-large-v3 đa ngôn ngữ có chất lượng không đồng đều trên tiếng Việt game streaming:
- Clip 5 giây ngắn → Whisper kém tự tin hơn → tỷ lệ hallucination cao hơn
- Game audio nhiều SFX → VAD cần tune cẩn thận
- Code-switching Việt + Anh gaming slang → Whisper xử lý OK nhưng không tối ưu

**Hai backend được hỗ trợ:**

| Backend | Model | Ưu điểm | Nhược điểm |
|---------|-------|---------|------------|
| `whisper` | `openai/whisper-large-v3` | Code-switching Việt+Anh tốt, faster-whisper nhanh | Đa ngôn ngữ → tiếng Việt không specialized |
| `phowhisper` | `vinai/PhoWhisper-large` | Fine-tuned trên tiếng Việt lớn, accent tốt hơn | Yếu hơn với English gaming terms |

**Chọn backend trong `config.yaml`:**
```yaml
annotation:
  asr:
    backend: "whisper"   # "whisper" | "phowhisper"  ← đổi tại đây

    whisper:
      model_name: "openai/whisper-large-v3"
      compute_type: "int8_float16"   # nhanh hơn float16, quality gần như nhau
      language: "vi"
      vad_filter: true
      no_speech_threshold: 0.45     # 0.45 thay vì default 0.6 — phù hợp clip 5s
      beam_size: 5
      condition_on_previous_text: false  # tránh repetition loop

    phowhisper:
      model_name: "vinai/PhoWhisper-large"
      compute_type: "float16"
      chunk_length_s: 30
      batch_size: 8
      language: "vi"
```

**Gaming Initial Prompt** (chỉ áp dụng cho Whisper backend):
```python
_GAMING_INITIAL_PROMPT = (
    "Đây là livestream game của streamer Việt Nam. "
    "Streamer hay nói: GG, clutch, ace, headshot, MVP, noob, lag, buff, nerf, "
    "rank, bot, carry, feed, gank, roam, farm, push, die, kill, team, "
    "ơi trời, vãi, thôi rồi, ăn rồi, xong rồi, đi nào, vào nào."
)
```
Prompt bias tokenizer về gaming domain, giảm lỗi nhận dạng gaming slang.

**Factory `build_asr(cfg.asr)` trong `whisper_asr.py`:**
```python
from vie_gameemo.data.annotator.whisper_asr import build_asr

asr_inst, bartpho_inst = build_asr(cfg.asr)
asr_inst.load()
transcript = asr_inst.transcribe(audio_path)
asr_inst.unload()
```

### 6.2. BARTpho Post-Processing (Tùy chọn)

Optional step sau ASR để fix lỗi word boundary, thiếu dấu, và từ dính nhau — vấn đề phổ biến với game audio nhiễu.

```yaml
annotation:
  asr:
    bartpho:
      enabled: false             # true để bật (thêm ~3GB VRAM, ~5-10s/clip)
      model_name: "vinai/bartpho-syllable-1_5"
      max_length: 256
      num_beams: 4
      prefix: "Sửa lỗi chính tả và hoàn thiện câu: "
```

**Logic pipeline với BARTpho:**
```python
asr_inst, bartpho_inst = build_asr(cfg.asr)
asr_inst.load()
if bartpho_inst is not None:
    bartpho_inst.load()

transcript = asr_inst.transcribe(audio_path)
if bartpho_inst is not None and transcript:
    transcript = bartpho_inst.process(transcript)

asr_inst.unload()
if bartpho_inst is not None:
    bartpho_inst.unload()
```

**Safety guard trong `BARTphoPostProcessor.process()`:** Nếu output ngắn hơn 50% input → giữ nguyên original (tránh generation failure).

### 6.3. Text Encoder — Chọn qua config

Output của cả hai encoder đều có shape `(B, 1, 768)` → **drop-in swap**, không ảnh hưởng fusion module.

| Encoder | Model | Khi nào dùng |
|---------|-------|-------------|
| `xlmr` | `FacebookAI/xlm-roberta-base` | Default — code-switching Việt+Anh, gaming slang |
| `phobert` | `vinai/phobert-base-v2` | Transcript chủ yếu tiếng Việt thuần (e.g. dùng PhoWhisper) |

**Chọn encoder trong `config.yaml`:**
```yaml
text_encoder:
  type: "xlmr"      # "xlmr" | "phobert"  ← đổi tại đây

  xlmr:
    model_name: "FacebookAI/xlm-roberta-base"
    max_length: 128
    pooling: "cls"   # "cls" | "mean"

  phobert:
    model_name: "vinai/phobert-base-v2"   # hoặc vinai/phobert-large (1024d)
    max_length: 128
    pooling: "cls"
```

**Factory `build_text_encoder(cfg.text_encoder)` trong `text_xlmr.py`:**
```python
from vie_gameemo.encoders.text_xlmr import build_text_encoder

text_enc = build_text_encoder(cfg.text_encoder)
text_enc.to(device)
h_text = text_enc.encode(transcript)  # (1, 1, 768)
```

**XLMRTextEncoder** (`src/vie_gameemo/encoders/text_xlmr.py`):
```python
class XLMRTextEncoder(nn.Module):
    def __init__(self, model_name="FacebookAI/xlm-roberta-base",
                 max_length=128, pooling="cls", device="cuda"):
        # AutoTokenizer + AutoModel, frozen
        ...
    def encode(self, text: str) -> Tensor:   # → (1, 1, 768)
    def encode_batch(self, texts: list[str]) -> Tensor:  # → (B, 1, 768)
```

**PhoBERTTextEncoder** (`src/vie_gameemo/encoders/text_phobert.py`):
```python
class PhoBERTTextEncoder(nn.Module):
    def __init__(self, model_name="vinai/phobert-base-v2",
                 max_length=128, pooling="cls", device="cuda"):
        # AutoTokenizer + AutoModel, frozen
        ...
    def encode(self, text: str) -> Tensor:   # → (1, 1, 768)  (same as XLM-R)
    def encode_batch(self, texts: list[str]) -> Tensor:  # → (B, 1, 768)
```

### 6.4. Tổ hợp khuyến nghị (ASR + Text Encoder)

| Kịch bản | ASR Backend | BARTpho | Text Encoder | Lý do |
|----------|-------------|---------|--------------|-------|
| **Default (code-switching)** | `whisper` | off | `xlmr` | Xử lý tốt Việt+Anh, nhanh |
| **Vietnamese focus** | `phowhisper` | off | `phobert` | Cả hai specialized cho tiếng Việt |
| **Maximum quality** | `phowhisper` | **on** | `phobert` | Tốt nhất chất lượng, chậm nhất |
| **Fastest** | `whisper` (turbo) | off | `xlmr` | Dùng `whisper-large-v3-turbo` |

### 6.5. Thử nghiệm Stage 2c

| Ablation | Setup | Mục đích |
|----------|-------|---------|
| A | Whisper + XLM-R (baseline) | Pipeline cũ |
| B | PhoWhisper + XLM-R | ASR quality trên tiếng Việt |
| C | PhoWhisper + PhoBERT | Full Vietnamese specialization |
| D | PhoWhisper + BARTpho + PhoBERT | Maximum pipeline |
| E | No transcript (zeros) | Upper bound không có text |

---

## 7. Stage 3 — Conv-Attention Pre-Fusion (MỚI)

**Đây là thay đổi quan trọng nhất so với pipeline cũ.** Thay MULT bằng Conv-Attention module từ Emotion-LLaMAv2.

### 7.1. Lý do thay MULT bằng Conv-Attention

Theo paper Emotion-LLaMAv2 (Table 9):

| Pre-fusion | MER-UniBench | MMEVerse-Bench |
|---|---|---|
| No fusion | 77.43 | 65.13 |
| Q-Former (AffectGPT) | 77.95 (+0.52) | 65.85 (+0.72) |
| Attention only (AffectGPT) | 77.65 (+0.22) | 65.37 (+0.24) |
| **Conv only (ours)** | 77.93 (+0.50) | 65.92 (+0.79) |
| **Conv-Attention (ours)** | **78.91 (+1.48)** | **66.05 (+0.92)** |

→ Conv-Attention **+1.48% trên SOTA benchmark**, hiệu quả hơn cả Q-Former.

### 7.2. Kiến trúc Conv-Attention (4 Modality)

**Lưu ý quan trọng:** Pipeline cũ Emotion-LLaMAv2 dùng 3 modality (audio/visual-global/visual-temporal). Pipeline của ta có **4 modality** vì tách visual thành 2 path (face + context):

```
4 Modality input:
  u_audio:  [B, 64, 768]    audio sequence (64 tokens after adaptive pool)
  u_face:   [B, T_f, 768]   streamer face from Path 1
  u_ctx:    [B, T_c, 768]   gameplay context from Path 2
  u_text:   [B, T_t, 768]   text from XLM-R

Standardize channels via modality-specific MLP, align seq lengths, then:
  F_d = Concat([MLP(u_audio), MLP(u_face), MLP(u_ctx), MLP(u_text)], dim=channel)
        shape: [B, seq_len, 4*768]
  F_s = Stack([MLP(u_audio), MLP(u_face), MLP(u_ctx), MLP(u_text)], dim=last)
        shape: [B, seq_len, 768, 4]

┌─────────────────────────────────────────────────────────────────┐
│                    Conv-Attention Module (4-modality)           │
│                                                                  │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │ Conv Branch     │         │ Attention Branch│               │
│   │ (local pattern) │         │ (modality       │               │
│   │                 │         │  weighting)     │               │
│   │ F_conv_0 = F_d  │         │ Attn_MLP(F_d)   │               │
│   │   ↓             │         │   → 4 weights   │               │
│   │ for k=1..N:     │         │   ↓             │               │
│   │   Conv1d        │         │ Softmax         │               │
│   │   Switch act    │         │   × F_s         │               │
│   │   Residual      │         │   = F_attn      │               │
│   │   ↓             │         │                 │               │
│   │ F_conv (final)  │         │                 │               │
│   └────────┬────────┘         └────────┬────────┘               │
│            │                           │                         │
│            └───────────┬───────────────┘                         │
│                        │                                         │
│                  F_conv + F_attn                                 │
│                        │                                         │
│                   u_fusion (768)                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Insight về tham số `n_modalities=4`:**
- Conv branch: input dim tăng từ 3*768=2304 lên **4*768=3072**
- Attention branch: softmax weighting trên **4 modality** thay vì 3
- Cho phép model học weight động giữa face/context — vd: cutscene cinematic → tăng weight context, gameplay action → tăng weight face

### 7.3. Implementation PyTorch (4 Modality)

```python
import torch
import torch.nn as nn

class SwitchActivation(nn.Module):
    """Switch activation từ paper Emotion-LLaMAv2"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)  # SiLU/Swish-like


class ConvBranch(nn.Module):
    """Convolutional branch — capture local patterns"""
    def __init__(self, in_dim=768*4, hidden_dim=768, n_blocks=4):
        super().__init__()
        self.initial_conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                SwitchActivation(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                SwitchActivation(),
            ))
    
    def forward(self, F_d):
        # F_d shape: [B, seq_len, 4*768]
        x = F_d.transpose(1, 2)  # [B, 4*768, seq_len]
        x = self.initial_conv(x)  # [B, 768, seq_len]
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # residual connection
        
        return x.transpose(1, 2)  # [B, seq_len, 768]


class AttentionBranch(nn.Module):
    """Attention branch — modality weighting"""
    def __init__(self, in_dim=768*4, n_modalities=4):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, n_modalities)  # Output 4 weights
        )
    
    def forward(self, F_d, F_s):
        # F_d: [B, seq, 4*768], F_s: [B, seq, 768, 4]
        attn_weights = self.attn_mlp(F_d)  # [B, seq, 4]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.unsqueeze(-2)  # [B, seq, 1, 4]
        
        F_attn = (F_s * attn_weights).sum(dim=-1)  # [B, seq, 768]
        return F_attn, attn_weights  # return weights for visualization


class ConvAttentionModule4M(nn.Module):
    """
    Conv-Attention pre-fusion cho 4 modality.
    Adapted từ Emotion-LLaMAv2 cho dual-path visual + audio + text.
    """
    def __init__(self, dim=768, n_modalities=4, n_blocks=4):
        super().__init__()
        # MLPs standardize từng modality
        self.mlp_audio = nn.Linear(dim, dim)
        self.mlp_face = nn.Linear(dim, dim)
        self.mlp_ctx = nn.Linear(dim, dim)
        self.mlp_text = nn.Linear(dim, dim)
        
        # Two branches
        self.conv_branch = ConvBranch(
            in_dim=dim*n_modalities, hidden_dim=dim, n_blocks=n_blocks
        )
        self.attn_branch = AttentionBranch(
            in_dim=dim*n_modalities, n_modalities=n_modalities
        )
    
    def forward(self, u_audio, u_face, u_ctx, u_text, has_face=True):
        """
        u_audio: [B, T_a, 768]  - audio sequence
        u_face:  [B, T_f, 768]  - face (path 1)
        u_ctx:   [B, T_c, 768]  - context (path 2)
        u_text:  [B, T_t, 768]  - text
        has_face: bool flag - False khi no-facecam clip
        """
        # Align all sequences to same length T
        T = u_audio.shape[1]  # use audio length as reference
        
        # Resample/broadcast to T
        u_face_aligned = self._align_to_T(u_face, T)
        u_ctx_aligned = self._align_to_T(u_ctx, T)
        u_text_aligned = self._align_to_T(u_text, T)
        
        # Modality dropout: nếu no-facecam, zero out face modality
        if not has_face:
            u_face_aligned = torch.zeros_like(u_face_aligned)
        
        # MLP standardize
        u_a = self.mlp_audio(u_audio)
        u_f = self.mlp_face(u_face_aligned)
        u_c = self.mlp_ctx(u_ctx_aligned)
        u_t = self.mlp_text(u_text_aligned)
        
        # Hybrid structures
        F_d = torch.cat([u_a, u_f, u_c, u_t], dim=-1)  # [B, T, 4*768]
        F_s = torch.stack([u_a, u_f, u_c, u_t], dim=-1)  # [B, T, 768, 4]
        
        # Two branches
        F_conv = self.conv_branch(F_d)
        F_attn, attn_weights = self.attn_branch(F_d, F_s)
        
        # Combine
        u_fusion = F_conv + F_attn  # [B, T, 768]
        return u_fusion, attn_weights  # return weights for interpretability
    
    def _align_to_T(self, u, T):
        """Align sequence length to T via broadcast/interpolate"""
        if u.shape[1] == T:
            return u
        elif u.shape[1] == 1:
            # broadcast single vector
            return u.expand(-1, T, -1)
        else:
            # interpolate
            return nn.functional.interpolate(
                u.transpose(1, 2), size=T, mode='linear', align_corners=False
            ).transpose(1, 2)
```

**Lợi ích visualization từ `attn_weights`:**
- Plot weight của từng modality theo timeline → biết model focus đâu khi nào
- Vd: cutscene → weight context cao; combat → weight face + audio cao
- Đây là **interpretability bonus** đặc thù cho domain game

### 7.4. So Sánh Các Pre-Fusion Strategies (Ablation)

Vẫn ablate so sánh, **nhưng giờ là 4 modality**:

| Pre-fusion | Implementation | Expected gain |
|---|---|---|
| **No fusion (Late)** | Voting | Baseline |
| **Early fusion** | Concat 4 modality + MLP | +0.5% |
| **MULT (cross-modal Transformer)** | Pipeline cũ, adapt 4 modality | +1.0% |
| **Q-Former** | AffectGPT-style | +0.7% |
| **Conv only** | Just conv branch | +0.5% |
| **Attention only** | Just attn branch | +0.2% |
| **Conv-Attention 4M (recommend)** | Cả 2 branch, 4 modality | **+1.5-2.0%** |

### 7.5. Hyperparameters

| Tham số | Giá trị | Lý do |
|---|---|---|
| `d_model` | 768 | Match all encoders |
| `n_modalities` | **4** | audio/face/context/text |
| `n_conv_blocks` | 4 | Paper recommendation |
| `kernel_size` | 3 | Standard, local pattern |
| Activation | Switch (x * sigmoid(x)) | Paper choice |
| Initial residual | Yes | Stable training |

---

## 8. Stage 4 — Emotion Classifier

Giữ nguyên như pipeline cũ.

```python
class EmotionClassifier(nn.Module):
    def __init__(self, dim=768, hidden=256, n_classes=9, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    
    def forward(self, u_fusion):
        # Pool u_fusion sequence to single vector
        h = u_fusion.mean(dim=1)  # [B, 768]
        return self.net(h)
```

**Loss:** Focal Loss cho class imbalance.

---

## 9. Stage 5 — LLM Reasoner (4 Setup, Có RLVR)

### 9.1. Tổng quan 4 setup

| Setup | Vai trò LLM | Training | Compute |
|---|---|---|---|
| **LLM-1: Post-hoc Explainer** | Giải thích sau prediction | No training | Lowest |
| **LLM-2: Co-Reasoner** | Tham gia inference (modality-to-text) | Optional SFT | Low |
| **LLM-3: VLM End-to-End** | Backbone đa phương thức | LoRA fine-tune | High |
| **LLM-4: RLVR-trained (MỚI)** | Reinforcement learning | Cold start + GRPO | Highest |

### 9.2. Setup LLM-1 — Post-hoc Explainer

Giữ nguyên như pipeline cũ. Qwen2.5-7B q4 nhận label + features → sinh giải thích VN.

### 9.3. Setup LLM-2 — Co-Reasoner

Modality-to-text + LLM tổng hợp. Có thể optional fine-tune với reasoning data từ multi-agent.

### 9.4. Setup LLM-3 — VLM End-to-End

Qwen2.5-VL-7B + LoRA fine-tune trên Vie-GameEmo.

### 9.5. Setup LLM-4 — RLVR-Trained (MỚI)

**Đây là setup mới nhất, đóng góp paper-worthy.** Áp dụng kỹ thuật từ R1-Omni cho VN domain.

#### Pipeline RLVR

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Cold Start (SFT trên 50-100 clip có reasoning)      │
│                                                              │
│  - Dùng reasoning từ multi-agent annotation                  │
│  - Train base LLM với supervised fine-tuning ngắn (1-2 epoch)│
│  - Mục tiêu: làm quen với format <think>...</think>          │
│              <answer>...</answer>                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: RLVR với GRPO trên full dataset                     │
│                                                              │
│  Reward function: R = R_acc + R_format                       │
│    R_acc = 1 if predicted emotion = ground truth else 0      │
│    R_format = 1 if output match <think>...<answer>... else 0 │
│                                                              │
│  GRPO:                                                       │
│    - Generate G=4 responses per question                     │
│    - Compute reward for each                                 │
│    - Normalize: A_i = (r_i - mean(r)) / std(r)               │
│    - Policy update với KL constraint                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Evaluation                                          │
│                                                              │
│  Test in-distribution: Vie-GameEmo test split                │
│  Test out-of-distribution: clip từ kênh streamer mới         │
│                            hoặc dataset emotion VN khác       │
│                                                              │
│  So sánh với SFT-only baseline:                              │
│    Expected: RLVR > SFT gap lớn trên OOD (~10-15%)           │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation với TRL library

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Base model — chọn theo compute available
# Option A: Có A100 → Qwen2.5-7B (better accuracy)
# Option B: Chỉ có Colab → Qwen2.5-0.5B hoặc HumanOmni-0.5B
model_name = "Qwen/Qwen2.5-7B-Instruct"  # hoặc "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_4bit,  # 4-bit quantize
    device_map="auto"
)

# Reward function
def reward_function(samples, **kwargs):
    rewards = []
    for sample in samples:
        # Extract <answer>...</answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', sample['response'])
        ground_truth = sample['gt_emotion']
        
        # R_acc
        if answer_match:
            predicted = answer_match.group(1).strip().lower()
            r_acc = 1.0 if predicted == ground_truth.lower() else 0.0
        else:
            r_acc = 0.0
        
        # R_format: check both <think> and <answer> tags
        format_ok = (
            '<think>' in sample['response'] and 
            '</think>' in sample['response'] and
            '<answer>' in sample['response'] and 
            '</answer>' in sample['response']
        )
        r_format = 1.0 if format_ok else 0.0
        
        rewards.append(r_acc + r_format)
    return rewards


# GRPO config
config = GRPOConfig(
    output_dir="./r1-vie-gameemo",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=4,  # G in GRPO
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=2,
    bf16=True,
    beta=0.04,  # KL coefficient
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    reward_funcs=[reward_function],
)
trainer.train()
```

#### Prompt cho RLVR

```python
PROMPT_RLVR = """Bạn là chuyên gia nhận diện cảm xúc. Xem video sau và phân tích 
cảm xúc của streamer.

Các nhãn cảm xúc có thể: {emotion_list}

Output theo format:
<think>
[Phân tích chi tiết: khuôn mặt, giọng nói, lời nói, bối cảnh, các bằng chứng đa 
phương thức dẫn đến kết luận]
</think>
<answer>[nhãn cảm xúc]</answer>
"""
```

### 9.6. Bảng So Sánh 4 Setup

| Aspect | LLM-1 | LLM-2 | LLM-3 | **LLM-4 (RLVR)** |
|---|---|---|---|---|
| Compute (train) | 0 | Low | High | **Highest** |
| Reasoning quality | Trung bình | Cao | Cao | **Cao + structured** |
| Classification accuracy | N/A | Trung bình | Cao | **Cao nhất kỳ vọng** |
| OOD generalization | N/A | Trung bình | Trung bình | **+10-15%** (R1-Omni finding) |
| Explainability | Có | Cao | Cao | **Có + verifiable** |
| Cần A100? | Không | Không | Có (LoRA) | **Có hoặc 0.5B model** |

### 9.7. Câu Hỏi Nghiên Cứu (Cập nhật)

> **RQ1:** Trong 4 setup LLM, setup nào cho classification accuracy cao nhất?
>
> **RQ2:** Setup nào có human eval score cao nhất về explanation quality?
>
> **RQ3:** RLVR có thực sự generalize tốt hơn SFT trên OOD data trong domain VN không? Gap bao nhiêu?
>
> **RQ4:** RLVR có giảm hallucination trong reasoning không? (R1-Omni paper báo cáo hallucination vẫn còn)

---

## 10. Perception-to-Cognition Curriculum Training (MỚI)

**Lấy ý tưởng từ Emotion-LLaMAv2.** Thay vì train classifier + LLM cùng lúc, chia thành 2 stage.

### 10.1. Lý do dùng Curriculum

Theo paper Emotion-LLaMAv2 (Table 10):

| Training scheme | MER-UniBench | MMEVerse-Bench |
|---|---|---|
| Single joint training | 75.54 | 64.20 |
| Perception-only (Stage 1) | 78.91 | 66.05 |
| Full Perception → Cognition | 78.52 | **66.63** |

→ Curriculum **vượt 3.4% so với joint training** trên MER-UniBench.

### 10.2. 2-Stage Training Schedule

#### Stage 1 — Perception (Recognition Only)

**Mục tiêu:** Align multimodal embeddings với emotion label space.

**Training data:** Tất cả clip với chỉ emotion label, không cần reasoning.

**Loss:** Cross-entropy hoặc Focal Loss cho classifier.

**Training:**
- Frozen encoders (AST, ViT, XLM-R)
- Trainable: Conv-Attention + Classifier MLP
- LR: 2e-4
- Epochs: 30
- Batch: 16 (effective với grad accumulation)

**Endpoint:** Emotion-LLaMAv2 paper báo cáo accuracy ~78% trên MER-UniBench sau Stage 1.

#### Stage 2 — Cognition (Joint Recognition + Reasoning)

**Mục tiêu:** Học LLM sinh reasoning trong khi vẫn giữ accuracy classification.

**Training data:** Subset clip có reasoning annotation (từ multi-agent pipeline).

**Loss:** Multi-task:
```
L = α * L_classification + β * L_reasoning_LM
```
trong đó:
- `L_classification`: cross-entropy của classifier (giữ accuracy)
- `L_reasoning_LM`: language modeling loss của LLM trên reasoning text
- α, β: weights (recommend α=1.0, β=0.5)

**Training:**
- Encoders + Conv-Attention frozen (đã train Stage 1)
- Trainable: LLM adapter + LoRA (rank=16, alpha=32)
- LR: 2e-5 (10x lower vì fine-tune)
- Epochs: 10-15
- Batch: 4-8

### 10.3. Implementation Skeleton

```python
class EmotionLLaMAv2Style(nn.Module):
    """Pipeline với curriculum support — DUAL-PATH VISUAL"""
    def __init__(self):
        super().__init__()
        # Frozen encoders
        self.audio_encoder = ASTModel.from_pretrained("MIT/ast-...")
        self.face_encoder = StreamerFaceEncoder()       # Path 1
        self.context_encoder = GameContextEncoder()     # Path 2
        self.text_encoder = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
        
        for p in self.audio_encoder.parameters(): p.requires_grad = False
        for p in self.face_encoder.parameters(): p.requires_grad = False
        for p in self.context_encoder.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters(): p.requires_grad = False
        
        # Trainable: 4-modality fusion
        self.fusion = ConvAttentionModule4M(dim=768, n_modalities=4)
        self.classifier = EmotionClassifier(dim=768, n_classes=9)
        
        # Webcam detector
        self.webcam_detector = WebcamDetector()
        
        # LLM (chỉ load ở Stage 2)
        self.llm = None
        self.llm_adapter = None
    
    def forward_perception(self, audio, frames, text, clip_path):
        """Stage 1 forward — chỉ classification (4 modality)"""
        # Audio
        h_audio = self.audio_encoder(audio)
        
        # Visual dual-path
        webcam_bbox = self.webcam_detector.detect_webcam_region(clip_path)
        if webcam_bbox is not None:
            face_crops = [extract_streamer_face(f, webcam_bbox) for f in frames]
            h_face_glo, h_face_temp = self.face_encoder(face_crops)
            has_face = True
        else:
            h_face_glo = torch.zeros(1, 1, 768)
            h_face_temp = torch.zeros(1, 1, 768)
            has_face = False
        
        h_face = (h_face_glo + h_face_temp) / 2  # combine global+temp
        h_ctx = self.context_encoder(frames)
        
        # Text
        h_text = self.text_encoder(text)
        
        # 4-modality fusion
        u_fusion, attn_weights = self.fusion(
            h_audio, h_face, h_ctx, h_text, has_face=has_face
        )
        logits = self.classifier(u_fusion)
        return logits, u_fusion, attn_weights
    
    def forward_cognition(self, audio, frames, text, clip_path, 
                          reasoning_target=None):
        """Stage 2 forward — classification + reasoning"""
        logits, u_fusion, _ = self.forward_perception(audio, frames, text, clip_path)
        
        if self.llm is not None:
            llm_input = self.llm_adapter(u_fusion)
            llm_output = self.llm(inputs_embeds=llm_input, labels=reasoning_target)
            return logits, llm_output.loss
        
        return logits, None


# === Stage 1 Training ===
model = EmotionLLaMAv2Style()
optimizer = torch.optim.AdamW([
    {'params': model.fusion.parameters(), 'lr': 2e-4},
    {'params': model.classifier.parameters(), 'lr': 2e-4},
])

for epoch in range(30):
    for batch in train_loader:
        logits, _, _ = model.forward_perception(
            batch['audio'], batch['frames'], batch['text'], batch['clip_path']
        )
        loss = focal_loss(logits, batch['label'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save checkpoint
torch.save(model.state_dict(), "checkpoint_stage1.pt")


# === Stage 2 Training ===
# Load LLM and adapter
model.llm = load_qwen_with_lora()
model.llm_adapter = nn.Linear(768, model.llm.config.hidden_size)

optimizer = torch.optim.AdamW([
    {'params': model.llm.parameters(), 'lr': 2e-5},
    {'params': model.llm_adapter.parameters(), 'lr': 2e-4},
    # Conv-Attention frozen sau Stage 1
])

for epoch in range(10):
    for batch in train_loader_with_reasoning:
        logits, lm_loss = model.forward_cognition(
            batch['audio'], batch['frames'], batch['text'],
            batch['clip_path'],
            reasoning_target=batch['reasoning_tokens']
        )
        cls_loss = focal_loss(logits, batch['label'])
        loss = 1.0 * cls_loss + 0.5 * lm_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 10.4. Ablation Curriculum

| Setup | Mô tả | Expected |
|---|---|---|
| Single joint | Train cả 2 nhiệm vụ cùng lúc | Baseline thấp |
| Perception only (Stage 1) | Chỉ classification | Mạnh về classification |
| Cognition only | Skip Stage 1, train Stage 2 trực tiếp | Yếu vì LLM không có grounding |
| **Full Curriculum** | Stage 1 → Stage 2 | **Tốt nhất** |

---

## 11. Chi Tiết Engineering Thực Tế

Giữ nguyên các tips từ pipeline cũ, **bổ sung** cho phần mới:

### 11.1. Memory cho Multi-Agent Annotation Pipeline

Multi-agent có nhiều model lớn chạy cùng → cần quản lý VRAM:

```python
# Bad: load tất cả model cùng lúc → OOM
vl_model = load_qwen_vl_72b()  # 40GB
audio_model = load_qwen_audio()  # 5GB
consolidator = load_qwen_32b()  # 20GB
# Total: 65GB → cần A100 80GB

# Good: pipeline serial, load/unload từng model
def annotate_clip_efficient(clip_path, emotion_label):
    # 1. Quick local steps (CPU/light GPU)
    peak_frame, au = detect_peak_frame_openface(clip_path)
    transcript = whisper_transcribe(clip_path)
    
    # 2. Load VL model, process, unload
    vl_model = load_qwen_vl_32b()
    Cvod = vl_model.generate(peak_frame, "...")
    del vl_model; torch.cuda.empty_cache()
    
    # 3. Load audio model, process, unload
    audio_model = load_qwen_audio()
    Catd = audio_model.generate(clip_path, "...")
    del audio_model; torch.cuda.empty_cache()
    
    # 4. Load consolidator, process, unload
    consolidator = load_qwen_32b()
    reasoning = consolidator.generate(prompt)
    del consolidator; torch.cuda.empty_cache()
    
    return {...}
```

**Trade-off:** chậm hơn (load model 30-60s mỗi clip) nhưng chạy được Kaggle T4 x2 (16GB x2).

**Đề xuất:** chạy batch — load 1 model → process 50-100 clip → unload → load model tiếp theo.

```python
# Optimal batching
def batch_annotate(clip_paths, emotion_labels):
    # Phase 1: Quick local (OpenFace + Whisper) cho tất cả clip
    quick_results = []
    for path in clip_paths:
        peak, au = detect_peak_frame_openface(path)
        ts = whisper_transcribe(path)
        quick_results.append({'peak': peak, 'au': au, 'transcript': ts})
    
    # Phase 2: Load VL model, batch process tất cả peak frames
    vl_model = load_qwen_vl_32b()
    cvods = vl_model.batch_generate([r['peak'] for r in quick_results])
    del vl_model; torch.cuda.empty_cache()
    
    # Phase 3: Audio model batch
    audio_model = load_qwen_audio()
    catds = audio_model.batch_generate(clip_paths)
    del audio_model; torch.cuda.empty_cache()
    
    # Phase 4: Consolidator batch
    consolidator = load_qwen_32b()
    reasonings = consolidator.batch_generate(prompts)
    del consolidator; torch.cuda.empty_cache()
    
    return [{...} for ...]
```

### 11.2. RLVR Training Tips

```python
# Critical: dùng vLLM cho fast generation trong GRPO
from trl import GRPOTrainer
from vllm import LLM, SamplingParams

# vLLM tăng tốc 5-10x cho generation step
vllm_model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    quantization="awq",  # nếu quantize
)

# GRPO config với vLLM
config = GRPOConfig(
    use_vllm=True,
    vllm_device="auto",
    ...
)
```

### 11.3. Curriculum Training Checkpoint Strategy

```python
# Stage 1 và Stage 2 lưu checkpoint khác nhau
checkpoint_stage1 = {
    'fusion': model.fusion.state_dict(),
    'classifier': model.classifier.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': 30,
    'stage': 1
}

checkpoint_stage2 = {
    'fusion': model.fusion.state_dict(),  # frozen sau S1
    'classifier': model.classifier.state_dict(),  # frozen sau S1
    'llm_adapter': model.llm_adapter.state_dict(),
    'llm_lora': get_peft_model_state_dict(model.llm),
    'epoch': 10,
    'stage': 2
}
```

### 11.4. Các tips giữ nguyên từ pipeline cũ

- Audio normalization per-clip (z-score)
- Face detection failure handling (zero pad hoặc repeat)
- Whisper hallucination với VAD filter
- Class imbalance với Focal Loss
- Caching features cho frozen encoders
- Reproducibility với seed

---

## 12. Thiết Kế Thử Nghiệm Tổng Thể

### 12.1. Bảng Ablation Mở Rộng (18+ Experiments)

| Ablation | Stage | Setup | Mục đích |
|---|---|---|---|
| **Modality ablation** | 3 | A/F/C/T riêng và combination | Đóng góp từng modal trong 4-modality |
| **Schema** | 0 | Ekman vs Gaming-7 | Schema tốt hơn |
| **Annotation** | 0 | Multi-agent vs Manual | Quality gap |
| **Consolidator model** | 0 | Qwen2.5-32B vs 72B | Cần model lớn không |
| **Audio encoder** | 2a | AST vs Wav2Vec2 vs HuBERT | Best for game audio |
| **Audio tokens** | 2a | 1/2/4/8/16/32/64/128 | Sweet spot |
| **Visual strategy (KEY)** | 2b | A: Full-frame / B: Face only / **C: Dual-path** | **Validate dual-path tốt nhất cho domain** |
| **Per-genre breakdown (KEY)** | 2b | A/B/C × {MOBA, FPS, Horror, RPG, Casual} | **Show genre-specific issues của full-frame** |
| **Webcam detection accuracy** | 2b | Đo trên 600 clip | Validate detector |
| **No-facecam handling** | 2b | Subset không có webcam | Test fallback |
| **Visual encoder pairing** | 2b | 5 combinations face/context encoder | Best synergy |
| **Number of frames** | 2b | 1/2/4/8/16/32/64 | Sweet spot |
| **Spatial pooling** | 2b | 1×1/2×2/4×4/8×8 | Best granularity |
| **Face crop margin** | 2b | 0.1/0.2/0.3 | Bao nhiêu context quanh face |
| **ASR** | 2c | Whisper-v3 vs PhoWhisper | Code-switching |
| **Text encoder** | 2c | XLM-R vs PhoBERT-v2 | Encoder phù hợp |
| **Pre-fusion** | 3 | Late/Early/MULT/Q-Former/Conv/Attn/**Conv-Attn 4M** | **Validate Conv-Attn best với 4 modality** |
| **Modality dropout** | 3 | p=0 vs 0.1 vs 0.3 | Robustness, no-facecam handling |
| **Attention weight visualization** | 3 | Plot weights theo timeline + genre | Interpretability: model focus đâu khi nào |
| **Loss** | 4 | CE vs Weighted CE vs Focal | Imbalance |
| **Training scheme** | All | Single vs Curriculum | **Validate curriculum** |
| **LLM setup** | 5 | LLM-1 vs LLM-2 vs LLM-3 vs **LLM-4** | **Vai trò LLM, RLVR** |
| **LLM choice** | 5 | Qwen2.5 vs SeaLLM vs Vistral | Best VN LLM |
| **RLVR vs SFT** | 5 | RLVR LLM-4 vs SFT-only | **Generalization gap** |
| **OOD generalization** | 5 | Test trên kênh streamer mới | **R1-Omni-style eval** |

### 12.2. Metrics

**Quantitative (Stage 4):**
- Accuracy, Weighted-F1 (WAF), Unweighted Average Recall (UAR)
- Per-class F1
- Confusion matrix

**Reasoning (Stage 5):**
- **Clue Overlap, Label Overlap** scores (EMER-style, dùng GPT-4o làm judge HOẶC dùng Qwen2.5-72B làm judge để 100% open-source)
- Human eval Likert 1-5: chính xác, thuyết phục, chi tiết
- Cohen's κ giữa 3 reviewer

**Generalization (R1-Omni-inspired):**
- Train trên Vie-GameEmo, test trên:
  - In-distribution: Vie-GameEmo test split
  - **Out-of-distribution**: clip từ streamer chưa thấy trong train
  - Cross-genre: train trên MOBA+FPS, test trên Horror+RPG

**Efficiency:**
- Inference time per clip (batch và real-time mode)
- VRAM peak
- Training time mỗi stage

### 12.3. Statistical Significance

- Báo cáo mean ± std qua **3 random seeds**
- Paired t-test khi so sánh 2 setup trên cùng test set
- Bootstrap confidence intervals cho human eval

### 12.4. Split Dataset

```
Total: 600 clip
Train:      70% (420 clip)
Validation: 15% (90 clip)
Test (ID):  10% (60 clip)
Test (OOD): 5% (30 clip)  ← streamer mới, không thấy ở train

Stratified theo: class, genre, code-switching status
```

---

## 13. Timeline 14 Tuần Chi Tiết

### Phân chia vai trò

- **Người A — "Vision & Audio Lead":** face, spectrogram, audio encoder, **Conv-Attention module**, real-time inference.
- **Người B — "Language & LLM Lead":** ASR, text encoder, **multi-agent annotation**, LLM reasoner (4 setup, **bao gồm RLVR**), demo.
- **Chia sẻ:** annotation verify, fusion debug, ablation, báo cáo.

### Lịch trình

| Tuần | Người A | Người B | Deliverable |
|---|---|---|---|
| **1** | Đọc paper (Emotion-LLaMAv2, R1-Omni), setup repo | Đọc paper, setup multi-agent infra | Survey doc + repo |
| **2** | Crawl 100 clip, tách face/audio | **Setup Whisper + Qwen-VL/Audio/72B** | 100 clip raw + agent test |
| **3** | EDA visual + audio, **đo % face dominant theo genre** | EDA text + **Test multi-agent pipeline trên 20 clip** | EDA report + Pilot annotation |
| **4** | Annotation verify pilot 50 clip | **Run multi-agent annotation 400 clip** | 400 clip có label + reasoning |
| **5** | Stage 2a (Spectrogram + AST) | Stage 2c (XLM-R baseline) | 2 unimodal baseline |
| **6** | **Stage 2b: WebcamDetector + Strategy B (face crop)** | Continue annotation → 600 clip | Face crop baseline + dataset full |
| **7** | **Stage 2b: Strategy C (dual-path) + Conv-Attn 4M** | Setup LLM-1 (Qwen2.5-7B prompt eng) | Dual-path + Conv-Attn + LLM-1 |
| **8** | **Stage 1 Perception training (dual-path)** | LLM-1 eval + LLM-2 setup | Stage 1 model + LLM-1/2 |
| **9** | **Stage 2 Cognition training** | LLM-3 (Qwen2.5-VL LoRA) | Stage 2 + LLM-3 |
| **10** | Real-time pipeline (compression) | **LLM-4 RLVR cold start (50-100 clip)** | Realtime demo + cold start |
| **11** | **Strategy A vs B vs C ablation (KEY) + per-genre** | **LLM-4 RLVR full training** | **Strategy ablation table** + LLM-4 |
| **12** | Modality + granularity ablations + attn visualization | Human eval setup + chạy (50 clip × 4 LLM) | Ablation tables + Eval + viz |
| **13** | Demo Gradio batch + realtime | OOD generalization eval | Demo hoàn chỉnh |
| **14** | Slide + Method section | Báo cáo Intro/RW/Discussion | Bản nháp |
| **Buffer** | Refactor, GitHub | Hoàn thiện | Final |

### Critical path

**Tuần 4 (annotation 400 clip multi-agent):** đây là bottleneck. Nếu compute không đủ → giảm scope còn 200 clip.

**Tuần 6-7 (Dual-path implementation):** WebcamDetector phải work tốt trên data thật. Pilot test sớm tuần 3 để biết failure rate.

**Tuần 11 (RLVR training + Strategy ablation):** double critical path. Người A chạy 3 strategy (A/B/C) cùng 1 tuần — nặng. Có thể parallelize bằng cách chạy training overnight.

---

## 14. Phương Án Compute (Có vs Không Có A100)

Đây là phần quan trọng vì bạn chưa rõ có access A100 không. Tôi đề xuất 2 phương án song song.

### 14.1. Phương Án A — Có Access A100 (RunPod/Vast.ai ~$1.5/h)

**Ngân sách compute:** ~$50-100 cho toàn bộ đồ án.

**Tasks cần A100:**

| Task | Compute | Thời gian | Cost |
|---|---|---|---|
| Multi-agent annotation 600 clip (Qwen2.5-72B q4) | A100 80GB | 2-3 ngày | ~$70-100 |
| LLM-3 VLM LoRA fine-tune | A100 40GB | 12-24h | ~$20-40 |
| **LLM-4 RLVR training (Qwen2.5-7B)** | A100 80GB | 24-48h | ~$40-70 |
| **Total** | | | **~$130-210** |

**Compromise nếu budget hạn chế ($50-80):**
- Giảm multi-agent dùng Qwen2.5-32B q4 (chạy được 1×A100 40GB) → giảm ~50% cost
- LLM-4 RLVR dùng Qwen2.5-3B thay 7B → giảm ~30% time

### 14.2. Phương Án B — Không Có A100 (Chỉ Kaggle/Colab Free)

**Trade-off:** Phải dùng model nhỏ hơn, giảm scope.

**Adaptations:**

#### Cho Multi-Agent Annotation

| Component | Phương án A (A100) | Phương án B (Colab T4) |
|---|---|---|
| Visual objective (Cvod) | Qwen2.5-VL-72B q4 | **Qwen2.5-VL-7B** (15GB → fit T4) |
| Audio tone (Catd) | Qwen2-Audio-7B | Qwen2-Audio-7B q4 (5GB) |
| Consolidator (Cmd) | Qwen2.5-72B q4 | **Qwen2.5-7B-Instruct q4** (5GB) |

**Chất lượng giảm:** ~10-15% so với 72B, nhưng vẫn đủ tốt cho annotation.

**Workaround:** verify percentage cao hơn (50% thay vì 30%) để đảm bảo quality.

#### Cho RLVR

**Adaptation từ R1-Omni:** chính paper R1-Omni dùng **HumanOmni-0.5B** — model rất nhỏ. Bạn có thể replicate exactly:

| Model | Size | VRAM (4-bit) | Compute cần |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | 5GB | Cần A100 cho GRPO |
| **Qwen2.5-1.5B-Instruct** | 1.5B | 2GB | **Kaggle T4 đủ** |
| **Qwen2.5-0.5B-Instruct** | 0.5B | 1GB | **Colab free đủ** |

**Recommended:** Qwen2.5-1.5B-Instruct hoặc 0.5B làm base model cho RLVR khi không có A100.

**Expected performance:** RLVR vẫn cho gap với SFT trên OOD ngay cả với model nhỏ — đó là finding key của R1-Omni với 0.5B.

#### Bảng so sánh 2 phương án

| Aspect | Phương án A (A100) | Phương án B (Colab/Kaggle) |
|---|---|---|
| Multi-agent consolidator | Qwen2.5-72B | Qwen2.5-7B |
| Annotation quality | Gần GPT-4o | Kém 10-15% |
| Annotation throughput | 200 clip/ngày | 50-80 clip/ngày |
| LLM-3 VLM fine-tune | Qwen2.5-VL-7B full LoRA | Qwen2.5-VL-3B LoRA |
| **LLM-4 RLVR base** | **Qwen2.5-7B** | **Qwen2.5-1.5B hoặc 0.5B** |
| Total budget | $130-210 | $0 (free tier) |
| Timeline | Đúng 14 tuần | 14-16 tuần (chậm hơn) |
| Đóng góp paper | Potentially top-tier venue | Workshop hoặc local conference |

### 14.3. Decision Framework

```
IF (có budget $130-210):
    → Phương án A
    → Cho phép full Qwen2.5-7B cho RLVR
    → Annotation quality cao
    → Submission potential: top venue

ELIF (budget $50-80):
    → Phương án A hạn chế
    → Dùng Qwen2.5-32B annotation
    → RLVR với Qwen2.5-3B
    → Submission: medium venue

ELSE (budget $0):
    → Phương án B
    → 100% Colab/Kaggle free
    → RLVR với 0.5B-1.5B
    → Submission: workshop / domestic
```

### 14.4. Quyết định dần dần

**Tuần 1-3:** Không cần A100 — chỉ cần Colab/Kaggle cho EDA và annotation pilot.

**Tuần 4:** Quyết định phương án annotation. Nếu test multi-agent 7B chất lượng ok → dùng phương án B. Nếu chưa đủ → cân nhắc thuê A100 1 ngày để chạy 72B annotation.

**Tuần 10-11:** Quyết định cuối cho RLVR. Test trước với Qwen2.5-0.5B trên Colab — nếu thấy improvement đáng kể vs SFT → giữ option đó. Nếu muốn maximize performance → thuê A100 2 ngày cho Qwen2.5-7B.

---

## 15. Checklist Trước Khi Bắt Đầu

### 15.1. Setup môi trường (Tuần 1)

- [ ] Repo GitHub với cấu trúc đề xuất
- [ ] Kaggle/Colab notebook cho cả 2 thành viên
- [ ] Cài đặt môi trường:
```bash
# Core
pip install torch transformers peft bitsandbytes accelerate

# RLVR (mới)
pip install trl vllm

# Vision
pip install mediapipe opencv-python torchvision

# Audio
pip install librosa torchaudio faster-whisper demucs

# Multi-agent (cần GPU đủ lớn)
pip install transformers[torch]  # Qwen-VL, Qwen-Audio support

# Vietnamese NLP
pip install underthesea sentencepiece

# Annotation
pip install label-studio

# Demo
pip install gradio streamlit

# Tracking
pip install wandb

# Crawl
pip install yt-dlp chat-downloader
```

- [ ] Wandb account
- [ ] Google Drive cho cached features
- [ ] (Optional) RunPod/Vast.ai account nếu chọn phương án A

### 15.2. Đọc 7 paper bắt buộc

- [ ] **Emotion-LLaMAv2** — paper chính cho architecture
- [ ] **Emotion-LLaMA** — preliminary, để hiểu evolution
- [ ] **R1-Omni** — cho RLVR setup
- [ ] **MULT** — Tsai et al. ACL 2019 (so sánh với Conv-Attn)
- [ ] **AST** — Gong et al. 2021 (spectrogram encoder)
- [ ] **Whisper** — Radford et al. ICML 2023
- [ ] **MER2025 / AffectGPT** — competitor SOTA

### 15.3. Pilot Stage 0 (Tuần 2)

- [ ] Lập danh sách 50 video target
- [ ] Crawl 100 clip đầu tiên
- [ ] **Test multi-agent pipeline trên 20 clip** — đo:
  - Compute time per clip
  - Chất lượng reasoning output
  - VRAM peak
- [ ] So sánh Qwen2.5-72B vs 32B vs 7B consolidator
- [ ] Quyết định phương án annotation (A vs B)

### 15.4. Cấu trúc repo cập nhật

```
vie-gameemo-fullupgrade/
├── README.md
├── requirements.txt
├── configs/
│   ├── perception_stage.yaml
│   ├── cognition_stage.yaml
│   └── rlvr_stage.yaml
├── data/
│   ├── raw_videos/        # gitignore
│   ├── processed/
│   │   ├── audios/
│   │   ├── faces/
│   │   └── transcripts/
│   ├── annotations/
│   │   ├── schema_ekman.csv
│   │   ├── schema_gaming.csv
│   │   ├── reasoning_multi_agent.json  # MỚI
│   │   └── reasoning_human_verified.json  # MỚI
│   └── features/
├── src/
│   ├── crawl/
│   ├── preprocess/
│   ├── annotation/        # MỚI
│   │   ├── multi_agent.py
│   │   ├── peak_frame.py
│   │   ├── openface_au.py
│   │   ├── qwen_vl.py
│   │   ├── qwen_audio.py
│   │   └── consolidator.py
│   ├── models/
│   │   ├── audio_encoder.py        # AST
│   │   ├── visual/                  # MỚI — dual-path
│   │   │   ├── webcam_detector.py   # MỚI — detect webcam region
│   │   │   ├── face_encoder.py      # MỚI — Path 1: streamer face
│   │   │   └── context_encoder.py   # MỚI — Path 2: gameplay context
│   │   ├── text_encoder.py
│   │   ├── fusion/
│   │   │   ├── late.py
│   │   │   ├── early.py
│   │   │   ├── mult.py
│   │   │   ├── q_former.py
│   │   │   └── conv_attention_4m.py # MỚI — 4-modality Conv-Attn
│   │   ├── classifier.py
│   │   └── llm/
│   │       ├── explainer.py         # LLM-1
│   │       ├── reasoner.py          # LLM-2
│   │       ├── vlm.py               # LLM-3
│   │       └── rlvr.py              # MỚI — LLM-4
│   ├── train/
│   │   ├── perception.py            # MỚI — Stage 1
│   │   ├── cognition.py             # MỚI — Stage 2
│   │   ├── rlvr_grpo.py             # MỚI — RLVR
│   │   └── lora_finetune.py
│   ├── eval/
│   │   ├── classification.py
│   │   ├── reasoning.py
│   │   ├── human_eval.py
│   │   ├── strategy_abc_ablation.py # MỚI — Strategy A/B/C compare
│   │   ├── per_genre_eval.py        # MỚI — Per-genre breakdown
│   │   ├── attention_viz.py         # MỚI — modality weight viz
│   │   ├── ood_eval.py              # MỚI — generalization
│   │   └── efficiency.py
│   └── inference/
│       ├── batch.py
│       └── realtime.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_multi_agent_pilot.ipynb    # MỚI
│   ├── 03_webcam_detector_test.ipynb # MỚI — validate detector
│   ├── 04_baselines.ipynb
│   ├── 05_dual_path_strategy.ipynb   # MỚI — A vs B vs C
│   ├── 06_curriculum.ipynb           # MỚI
│   ├── 07_rlvr.ipynb                 # MỚI
│   └── 08_results.ipynb
├── demo/
│   ├── gradio_batch.py
│   └── streamlit_realtime.py
└── report/
```

### 15.5. Đóng Góp Có Thể Publish

Với pipeline này, có thể submit tới:
- **Workshop venues:** ACM MM workshops (Affective Computing), ICME workshops, NAACL workshops
- **Conference venues (nếu kết quả mạnh):** ACL Findings, EMNLP Findings, COLING, PACLING
- **Specialized venues:** ACII (Affective Computing), Interspeech (cho audio modality)

**Strongest contribution để emphasize:**
1. **Dual-path visual encoding for livestream gaming** — đầu tiên adapt MER cho domain ngoài talking-head, chỉ ra hạn chế của approach full-frame
2. **RLVR cho VN emotion domain** — đầu tiên, mới (R1-Omni Mar 2025)
3. **Multi-agent annotation 100% open-source cho VN**
4. **Conv-Attention 4-modality pre-fusion** validated trên VN game domain
5. **Curriculum P→C cho code-switching VN**

---

## Kết Luận

Pipeline Full Upgrade này tích hợp **7 hướng cải tiến** (6 từ paper + 1 domain adaptation):

1. ✅ **Dual-path visual encoding** (DOMAIN-SPECIFIC) — adapt cho livestream gaming, không follow paper máy móc
2. ✅ **Conv-Attention pre-fusion 4-modality** (thay MULT) — +1.5-2.0% accuracy
3. ✅ **Curriculum Perception→Cognition** — +3.4% so với single joint training
4. ✅ **RLVR LLM-4** — +10-15% OOD generalization
5. ✅ **Multi-agent annotation** (open-source) — giảm 70% effort, có reasoning data
6. ✅ **Encoder pairing ablation** — find best combination
7. ✅ **Granularity ablation** (audio tokens, frames, spatial pooling)

**Phương án compute linh hoạt:**
- Có A100 → full Qwen2.5-7B cho RLVR, Qwen2.5-72B cho annotation
- Không có A100 → Qwen2.5-1.5B/0.5B cho RLVR, Qwen2.5-7B cho annotation
- Pipeline scale-down vẫn giữ được core contributions

**Insight quan trọng về domain adaptation:**
- Không follow paper máy móc — đặc biệt khi domain khác nhau rõ rệt
- Emotion-LLaMAv2 work tốt vì data của họ là talking-head (60-85% face)
- Domain livestream game khác hoàn toàn (face streamer chỉ 10-15% frame, có face nhân vật game gây nhiễu)
- Dual-path là response domain-specific cho vấn đề này

**Team 2 người, 14 tuần** — feasible với careful planning, đặc biệt cần:
- Bắt đầu multi-agent annotation từ tuần 2
- Pilot WebcamDetector từ tuần 3 để đo failure rate
- Quyết định compute path ở tuần 4
- Critical path: multi-agent annotation (tuần 4), dual-path implementation (tuần 6-7), Strategy ablation + RLVR (tuần 11)

Đây là pipeline ambitious nhưng có nhiều **đóng góp paper-worthy** — nếu kết quả tốt, có thể submit tới venue chất lượng cao. Đặc biệt, **Strategy A vs B vs C ablation** là một contribution methodological mới: chứng minh follow paper máy móc không phải lúc nào cũng đúng cho domain mới.
