# Vấn đề: Dataset lẫn lộn video tiếng Việt và tiếng Anh trong pipeline phân loại cảm xúc multimodal

## 1. Bối cảnh dự án

**Vie-GameEmo** là đồ án học thuật (team 2 người, compute Kaggle/Colab + optional RunPod) xây dựng pipeline **multimodal emotion recognition** cho video game livestream/review tiếng Việt. Pipeline kết hợp 4 modality:

- **Audio**: spectrogram → AST encoder → `h_audio` (64, 768)
- **Visual face**: webcam crop → ViT-FER → `h_face` (1, 768)
- **Visual context**: full-frame gameplay → ViT → `h_ctx` (1, 768)
- **Text (transcript)**: ASR → text encoder → `h_text` (1, 768)

4 embedding được fuse bằng **Conv-Attention Pre-Fusion** (từ Emotion-LLaMAv2 2026) → classifier MLP → 8 nhãn cảm xúc gaming (`neutral, hype, amused, tilted, sad, shocked, fear, disgusted`).

Downstream có 4 setup LLM Reasoner (Explainer / Co-Reasoner / VLM / RLVR) nhận transcript như một input evidence, output reasoning bằng tiếng Việt.

**Dataset hiện tại**: ~3,245 clips × 5 giây (đang thu thập thêm), phân bố không cân bằng:

| Nhãn | Số lượng | % |
|------|----------|---|
| neutral | 1,396 | 43.0% |
| amused | 414 | 12.8% |
| hype | 379 | 11.7% |
| tilted | 295 | 9.1% |
| sad | 282 | 8.7% |
| fear | 263 | 8.1% |
| shocked | 200 | 6.2% |
| disgusted | 16 | 0.5% |

**Lưu ý**: `disgusted` đang rất thiếu (16 mẫu), đang thu thập thêm dự kiến đạt 100–150 mẫu. `neutral` chiếm tỷ lệ dominant (43%). Dataset có **class imbalance nghiêm trọng** — đây là thực tế tự nhiên của livestream gaming (phần lớn thời gian streamer ở trạng thái neutral).

## 2. Vấn đề cụ thể

**Source video tiếng Việt không đủ.** Livestream/review game tiếng Việt trên YouTube ít hơn nhiều so với tiếng Anh, đặc biệt ở một số genre (horror, RPG) và một số nhãn hiếm (`disgusted`, `fear`). Để đạt đủ số lượng mẫu cho các class thiếu, dataset phải bổ sung video tiếng Anh (streamer nói tiếng Anh).

Điều này tạo ra **dataset bilingual** — một phần clips có transcript tiếng Việt, một phần có transcript tiếng Anh — nhưng **pipeline hiện tại không xử lý được trường hợp video hoàn toàn tiếng Anh**.

## 3. Pipeline hiện tại xử lý được gì

### 3.1. Code-switching (streamer Việt nói xen tiếng Anh gaming) — ĐÃ XỬ LÝ

Pipeline đã có cơ chế tốt cho trường hợp **người Việt nói xen gaming slang tiếng Anh** (GG, clutch, ace, headshot...):

**ASR**: `WhisperASR` (faster-whisper) có gaming initial prompt bias:
```python
_GAMING_INITIAL_PROMPT = (
    "Đây là livestream game của streamer Việt Nam. "
    "Streamer hay nói: GG, clutch, ace, headshot, MVP, noob, lag, buff, nerf, "
    "rank, bot, carry, feed, gank, roam, farm, push, die, kill, team, "
    "ơi trời, vãi, thôi rồi, ăn rồi, xong rồi, đi nào, vào nào."
)
```

**Text encoder**: Default dùng `XLM-RoBERTa-base` — multilingual, xử lý code-switching Việt+Anh tốt.

**Schema**: `Annotation` có field `code_switching_ratio: float` để track tỷ lệ code-switching.

### 3.2. Video hoàn toàn tiếng Anh — CHƯA XỬ LÝ

Các điểm chặn (blocking issues):

#### a) ASR hardcode `language="vi"`

Cả hai ASR backend đều ép ngôn ngữ đầu ra là tiếng Việt:

```yaml
# config.yaml
annotation:
  asr:
    backend: "whisper"
    whisper:
      language: "vi"          # ← hardcode
    phowhisper:
      language: "vi"          # ← hardcode
```

```python
# whisper_asr.py — WhisperASR
class WhisperASR:
    def __init__(self, ..., language: str = "vi", ...):
        ...
    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,  # ← force "vi"
            initial_prompt=self.initial_prompt,  # ← Vietnamese gaming prompt
            ...
        )
```

**Hậu quả**: Video tiếng Anh sẽ cho transcript rác — Whisper bị ép transcribe tiếng Anh thành ký tự tiếng Việt, hoặc hallucinate nội dung không liên quan.

#### b) PhoWhisper không hỗ trợ tiếng Anh

`PhoWhisperASR` (vinai/PhoWhisper-large) chỉ fine-tuned cho tiếng Việt, hoàn toàn không transcribe được tiếng Anh.

#### c) BARTpho post-processing chỉ hoạt động với tiếng Việt

```python
# BARTphoPostProcessor
prefix: str = "Sửa lỗi chính tả và hoàn thiện câu: "  # Vietnamese instruction
```

#### d) Không có bước language detection

Không có cơ chế tự động phát hiện ngôn ngữ gốc của video/clip để route xử lý phù hợp.

#### e) Không có bước translation

Không có pipeline dịch transcript tiếng Anh → tiếng Việt (hoặc ngược lại).

#### f) LLM prompts và consolidator chỉ viết bằng tiếng Việt

Tất cả prompt trong multi-agent annotation pipeline và LLM Reasoner đều bằng tiếng Việt:

```python
# consolidator.py
CONSOLIDATION_PROMPT_TEMPLATE = """
Bạn là chuyên gia phân tích cảm xúc đa phương thức. Viết đoạn reasoning ngắn
(3-5 câu) bằng tiếng Việt, giải thích vì sao streamer này đang trong trạng
thái {emotion_label}.
...
- Lời nói (transcript): "{transcript}"    # ← transcript EN sẽ lạc lõng ở đây
"""

# llm2_coreasoner.py
_COREASONER_PROMPT = """
...
Lời nói (transcript): "{transcript}"      # ← tương tự
"""
```

#### g) Annotation schema thiếu trường ngôn ngữ

`Annotation` trong `schemas.py` không track ngôn ngữ gốc của clip:
```python
class Annotation(BaseModel):
    clip_id: str
    emotion_label: EmotionLabel
    transcript: str               # ← không biết đây là VI hay EN
    code_switching_ratio: float   # ← chỉ track code-switch, không track ngôn ngữ chính
    # THIẾU: source_language: str
```

## 4. Các modality khác KHÔNG bị ảnh hưởng

Quan trọng: chỉ có **text modality** bị ảnh hưởng. Các modality khác language-agnostic:

| Modality | Model | Ảnh hưởng? | Lý do |
|----------|-------|-----------|-------|
| Audio (spectrogram) | AST (MIT/ast-finetuned-audioset) | **Không** | Spectrogram features (pitch, energy, rhythm) không phụ thuộc ngôn ngữ |
| Face (visual) | ViT-FER (trpakov/vit-face-expression) | **Không** | Biểu cảm khuôn mặt universal |
| Context (visual) | ViT (google/vit-base-patch16-224) | **Không** | Gameplay screenshots không có text |
| **Text (transcript)** | XLM-R hoặc PhoBERT | **CÓ** | Phụ thuộc hoàn toàn vào ngôn ngữ transcript |

## 5. Các ràng buộc

- **Compute giới hạn**: Kaggle (2× T4 16GB) hoặc Colab (1× T4/L4). Không có A100 ngoại trừ optional RunPod.
- **Dataset thực tế ~3,245 clips** (lớn hơn target 600 ban đầu), class imbalance nặng (`neutral` 43%, `disgusted` 0.5%). Tỷ lệ VI/EN chưa rõ chính xác — video tiếng Anh chủ yếu được bổ sung cho các class hiếm (`disgusted`, `fear`, `shocked`) và genre ít video Việt (horror, RPG).
- **Đồ án học thuật**: cần giải thích rõ ràng lý do chọn approach, có ablation study.
- **Output cuối cùng phải bằng tiếng Việt**: LLM reasoning explanation phải bằng tiếng Việt.
- **XLM-R đã multilingual**: Text encoder default (XLM-R) có thể encode cả tiếng Anh lẫn tiếng Việt vào cùng embedding space. Đây là lợi thế sẵn có.
- **Whisper có khả năng tự detect ngôn ngữ**: Nếu bỏ param `language="vi"`, Whisper-large-v3 tự detect ngôn ngữ và transcribe đúng. Nó cũng hỗ trợ `task="translate"` để dịch bất kỳ ngôn ngữ nào sang tiếng Anh.
- **Cấu trúc config-driven**: Toàn bộ pipeline switch qua `config.yaml`, không cần sửa code khi đổi setup. Giải pháp nên giữ nguyên pattern này.

## 6. Các file liên quan trong codebase

```
src/vie_gameemo/
├── data/
│   ├── annotator/
│   │   ├── whisper_asr.py       # WhisperASR, PhoWhisperASR, BARTphoPostProcessor, build_asr()
│   │   ├── pipeline.py          # Multi-agent annotation orchestrator (annotate_batch)
│   │   └── consolidator.py      # Consolidator LLM (merge signals → reasoning)
│   ├── dataset.py               # PyTorch Dataset (loads cached features)
│   └── schemas.py               # Annotation, Clip, MultimodalFeatures schemas
├── encoders/
│   ├── text_xlmr.py             # XLMRTextEncoder + build_text_encoder() factory
│   └── text_phobert.py          # PhoBERTTextEncoder
├── llm/
│   ├── llm1_explainer.py        # Post-hoc explainer (nhận transcript)
│   ├── llm2_coreasoner.py       # Co-Reasoner (nhận transcript)
│   ├── llm3_vlm.py              # VLM end-to-end
│   └── llm4_rlvr.py             # RLVR-trained
├── fusion/                      # Conv-Attention Pre-Fusion (language-agnostic)
└── inference/
    ├── batch.py                 # Batch inference
    └── realtime.py              # Real-time inference

config.yaml                     # Central config (ASR backend, text encoder, etc.)
```

## 7. Câu hỏi cần giải quyết

1. **Cách tốt nhất để xử lý transcript từ video tiếng Anh** trong pipeline này là gì? (detect → translate → encode? detect → encode trực tiếp multilingual? hay approach khác?)

2. **Nên translate transcript EN→VI trước khi encode**, hay **giữ nguyên tiếng Anh và dựa vào XLM-R multilingual**? Trade-off giữa:
   - Translate: nhất quán ngôn ngữ, nhưng thêm model + latency + lỗi dịch
   - Giữ nguyên: đơn giản, nhưng embedding space có thể bị phân mảnh (VI cluster vs EN cluster)

3. **Ảnh hưởng đến LLM Reasoner**: Transcript tiếng Anh sẽ feed vào prompt tiếng Việt của LLM. Nên xử lý thế nào? Translate trước? Hay sửa prompt để bilingual?

4. **Ảnh hưởng đến annotation pipeline**: Consolidator prompt bằng tiếng Việt nhận transcript tiếng Anh. Có cần route riêng cho EN clips?

5. **Ảnh hưởng đến data split stratification**: Có nên thêm `source_language` vào stratification (cùng với class + genre + code-switching)?

6. **Ablation study**: Nên design thí nghiệm nào để đo impact của bilingual dataset? So sánh VI-only vs mixed vs translated?

7. **Có cách nào tận dụng được tín hiệu cảm xúc từ transcript bất kể ngôn ngữ** mà không phụ thuộc vào nội dung ngôn ngữ cụ thể? (ví dụ: sentiment-aware multilingual embeddings, emotion-specific features)
