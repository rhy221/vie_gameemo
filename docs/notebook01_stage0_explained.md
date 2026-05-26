# Phân tích Notebook 01 — Stage 0: Chuẩn bị Dataset

## Tổng quan

Notebook 01 là bước **xây dựng dataset có giám sát** cho toàn bộ pipeline Vie-GameEmo.
Output của nó là các file annotation JSON, mỗi file tương ứng với một clip 3–7 giây và chứa:

- Nhãn cảm xúc cuối cùng (ground truth cho training)
- Reasoning text tự động sinh (training signal cho LLM reasoner)
- Bằng chứng đa modal thô: transcript, AU, visual description, audio tone

---

## Tại sao pipeline này là "semi-automatic"?

Pipeline Vie-GameEmo không phải hoàn toàn tự động vì bản chất của bài toán:

| Thành phần | Người hay máy? | Lý do |
|---|---|---|
| Nhãn cảm xúc | **Người cung cấp** (qua `CLIP_LABELS`) | Ground truth cần phán đoán ngữ cảnh — máy không thể tự khởi động khi chưa có training data |
| Reasoning text | **Máy tự động sinh** | Consolidator giải thích "tại sao" dựa trên nhãn đã cho |
| Transcript | Máy (Whisper ASR) | — |
| Action Units | Máy (OpenFace/MediaPipe) | — |
| Visual description | Máy (Qwen-VL) | — |
| Audio tone | Máy (Qwen-Audio) | — |

**Bootstrapping problem**: Muốn train model tự gán nhãn, cần data có nhãn trước. Vòng đầu tiên
bắt buộc phải có nhãn người. Sau khi model đủ mạnh, có thể dùng nó để pseudo-label batch tiếp
theo (active learning / self-training).

### Vai trò của CLIP_LABELS

`CLIP_LABELS` trong CELL 2 là dict `clip_id → nhãn`:

```python
CLIP_LABELS = {
    'stream01_seg_000': 'hype',
    'stream01_seg_001': ('tilted', 0.85),   # với confidence
    'stream01_seg_002': {
        'label': 'amused',
        'confidence': 0.9,
        'alternatives': {'hype': 0.1}        # phân phối thay thế
    },
}
```

Nếu clip không có trong dict → nhãn mặc định `'neutral'` với `confidence=0.0` và
`is_placeholder=True`. Clip có `is_placeholder=True` **không nên dùng cho training thật**
vì nhãn không đáng tin cậy.

---

## Luồng dữ liệu từng bước

```
Video thô (YouTube / file có sẵn)
        │
        ▼ CELL 7 — ffmpeg stream-copy
Video clips 3–7 giây (data/clips/)
        │
        ├──────────────────────────────────────────────────────────┐
        ▼ CELL 8 — tách audio + frames                            │
  audio.wav (16kHz)   frames/ (2fps, JPEG)                        │
        │                     │                                    │
        │              ┌──────┘                                    │
        │              │                                           │
        ▼              ▼                                           ▼
CELL 10: Whisper   CELL 11: OpenFace   CELL 12: Qwen-VL    CELL 9: DBSCAN
transcript          face_aus             visual_objective    webcam bbox
        │              │                      │
        └──────────────┴──────────────────────┘
                       │
               CELL 13: Qwen-Audio
                  audio_tone
                       │
                       ▼
              CELL 14: Consolidator
                    ┌─────────────────────────────┐
                    │ Input: transcript + face_aus │
                    │        + visual + audio tone │
                    │        + manual_label (anchor)│
                    │                             │
                    │ Output: reasoning (3-5 câu) │
                    │         pred_label          │
                    │         pred_confidence     │
                    │         distribution TOP-3  │
                    └─────────────────────────────┘
                       │
                       ▼
              CELL 15: Tổng hợp → annotation JSON
```

---

## Cấu trúc Annotation JSON

Mỗi clip tạo ra một file `{clip_id}.json` với cấu trúc:

```json
{
  "clip_id": "stream01_seg_042",
  "video_path": "/kaggle/working/.../stream01_seg_042.mp4",

  "segment": {
    "raw_video": "stream01.mp4",
    "start_sec": 210.0,
    "duration_sec": 5.0
  },

  "emotion_label": "hype",
  "confidence": 0.87,
  "label_source": "model",

  "manual_label": {
    "label": "hype",
    "confidence": 0.9,
    "alternatives": {},
    "is_placeholder": false
  },

  "predicted_label": {
    "label": "hype",
    "confidence": 0.87,
    "distribution": { "hype": 0.75, "amused": 0.15, "neutral": 0.10 }
  },

  "transcript": "Ôi trời ơi vô rồi, vô rồi! ...",
  "face_aus": "AU12=4.1 AU6=3.8 AU25=5.0 ...",
  "visual_objective": "Streamer ngồi trước màn hình, miệng mở rộng ...",
  "audio_tone": "Giọng nói to, tốc độ nhanh, tần số cao ...",
  "reasoning": "Streamer hét lên khi thấy kill streak. Nụ cười rộng (AU12=4.1) ...",

  "annotation_model": "Qwen/Qwen2.5-7B-Instruct",
  "created_at": "2026-05-26T14:30:00",
  "split": "train"
}
```

### Từng trường được dùng ở đâu

| Trường | Dùng trong | Mục đích |
|--------|-----------|----------|
| `emotion_label` | Notebook 02 — perception training | Classification target (9 classes) |
| `emotion_label` | Notebook 02 — cognition/RLVR | `<answer>` trong `<think>...</think><answer>hype</answer>` |
| `reasoning` | Notebook 02 — SFT cold-start | Supervised target cho LLM reasoner |
| `transcript` | Notebook 02 — XLM-R encoder | Text modality feature |
| `face_aus` | LLM evidence dict | Context cho LLM khi giải thích |
| `visual_objective` | LLM evidence dict | Context cho LLM khi giải thích |
| `audio_tone` | LLM evidence dict | Context cho LLM khi giải thích |
| `confidence` | Lọc data chất lượng thấp | Loại clip có confidence < 0.5 |
| `manual_label.is_placeholder` | Lọc data | Loại clip chưa được gán nhãn thực |
| `predicted_label.distribution` | Soft label training | Có thể dùng label smoothing |

---

## Nhãn cuối cùng: Model hay Human?

CELL 15 hiện tại đặt `label_source = 'model'`, nghĩa là **nhãn cuối cùng luôn là output của Consolidator**,
không phải nhãn thủ công:

```python
final_label = pred['label']        # từ Consolidator
final_confidence = pred['confidence']
label_source = 'model'
```

### Tại sao lại để model quyết định?

1. **Consolidator có đầy đủ context**: nó thấy transcript + AU + visual + audio trước khi quyết định.
   Người annotate chỉ xem video, Consolidator phân tích 4 chiều cùng lúc.
2. **Nhãn thủ công vẫn có giá trị**: Consolidator được cho xem `manual_label` trong prompt
   như một "gợi ý". Nếu Consolidator đồng ý → confidence cao. Nếu không đồng ý → phát hiện
   edge case cần review.
3. **Consistency**: cùng chuẩn model, không phụ thuộc vào người annotate khác nhau.

CELL 16 tính **agreement rate** (% clip model đồng ý với human) — nếu < 70% thì nên xem lại
guidelines hoặc CLIP_LABELS.

---

## Vai trò của Notebook 01 trong toàn pipeline

```
Notebook 01 (Stage 0)          Notebook 02 (Training)        Notebook 03 (Inference)
─────────────────────          ──────────────────────        ───────────────────────
annotation JSONs ──────────────► VieGameEmoDataset             (dùng model đã train)
                                 ├─ emotion_label → Perception
                                 │   (fusion + classifier)
                                 └─ reasoning → Cognition
                                     (SFT + RLVR LLM)
```

Notebook 01 không phải chỉ tạo data cho một mục tiêu — nó tạo **toàn bộ training corpus** cho cả
hai model: perception (classification) và cognition (reasoning generation).

---

## Khi nào có thể bỏ Notebook 01?

- Khi đã có annotation dataset sẵn trên Kaggle (mount vào Notebook 02/03 trực tiếp).
- Khi dùng dataset bên ngoài đã có nhãn cảm xúc (cần convert sang format annotation JSON).
- Trong Notebook 03, nếu chỉ muốn inference thử nghiệm, notebook tự tạo dummy video
  để test pipeline mà không cần annotation nào.

---

## Lưu ý quan trọng khi chạy

| Vấn đề | Giải pháp |
|--------|-----------|
| OpenFace build fail | Tự động fallback sang MediaPipe (nhanh hơn, AU ít chính xác hơn) |
| CLIP_LABELS trống | Tất cả clip → `neutral` placeholder, **không dùng cho training** |
| VRAM T4 16GB | Mỗi model load/unload tuần tự — không load 2 model cùng lúc |
| Kaggle timeout 9h | Giới hạn `MAX_CLIPS` hoặc chạy nhiều session chia nhỏ dataset |
| Annotation chất lượng thấp | Lọc bằng `confidence >= 0.5` và `is_placeholder=False` trước khi train |
