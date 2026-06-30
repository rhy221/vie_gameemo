# Implementation Summary — Phase 0 + LLM-1 FIX 1–5

**Ngày:** 2026-06-30  
**Repo:** `vie-gameemo-skeleton`  
**Kết quả test:** 113/113 passed, 0 regression

---

## Tổng quan

Thực hiện theo `claude_code_prompt (1).md`:
- **Phase 0**: Thêm context encoder pose-kinematics song song ViT-ImageNet
- **FIX 1–5**: Sửa các lỗi trong LLM-1 faithful explainer

---

## FIX 3 — GHead global pool → GHeadPerModality

**File:** `src/vie_gameemo/training/llm1_explanation.py`

**Vấn đề:** `GHead` cũ dùng `_mean_pool_raw()` pool toàn cục 4 modality thành 1 vector → gradient loãng 1/4, modality mạnh che modality yếu, không đảm bảo từng raw token giữ cue riêng.

**Sửa:** Thay bằng `GHeadPerModality` — 4 head độc lập:

```
face_head  (768 → 128 → 5)   ← chỉ đọc face.mean(1)
voice_head (768 → 128 → 3)   ← chỉ đọc audio.mean(1)
motion_head(768 → 128 → 3)   ← chỉ đọc context.mean(1), chỉ khi pose branch
text_head  (768 → 128 → 4)   ← chỉ đọc text.mean(1)
```

- `motion_head = None` khi `has_context=False` (nhánh `vit_imagenet`)
- `_compute_loss` split `attr_tensor` theo từng modality, cộng loss riêng
- GHead xác nhận không đọc `penult` hay `fusion` (chỉ tap A)

---

## FIX 4 — Bổ sung 2 eval faithfulness

**File:** `src/vie_gameemo/evaluation/faithfulness.py`

Thêm 2 hàm mới vào `evaluate_faithfulness()`:

### `eval_counterfactual()`
- Zero lần lượt từng modality input
- Nếu MLP đổi nhãn → generate explanation cả 2 lần (trước/sau perturb)
- Kiểm: LLM Emotion có đổi nhất quán theo MLP không?
- Metric: `counterfactual_consistency_rate = n_consistent / n_label_changed`
- "Stale" explanation (nhãn không đổi dù MLP đổi) = faithfulness failure

### `eval_hedge()`
- Trên tập gold, tìm các ca MLP sai (`mlp_pred != gold`)
- Kiểm confidence token Emotion tại generation step (dùng `output_scores=True`)
- Nếu top prob < 0.6 → coi là hedged (model giảm tự tin đúng cách)
- Metric: `hedge_rate = n_hedged_when_mlp_wrong / n_mlp_wrong`

---

## FIX 5 — Verification (không cần sửa code)

Đã kiểm tra trong code hiện có — tất cả đúng:

| Điểm kiểm | Kết quả |
|-----------|---------|
| KL direction: `F.kl_div(lm_soft, mlp_soft)` | `KL(teacher‖student)` ✓ |
| Penult eval mode: `classifier.eval()` trước khi freeze | Dropout off khi lấy penult ✓ |
| Face cue discretization: `_face_cue()` trả text | `"eyes=wide"`, `"mouth=open"` ✓ |
| Text leak: ghi chú trong faithfulness | Warning nếu Emotion đổi khi bỏ text token |

---

## Phase 0 — PoseContextEncoder (file mới)

**File:** `src/vie_gameemo/encoders/context_pose.py`

Context encoder mới thay ViT-ImageNet mean-pool (mù chuyển động) bằng pose kinematics + BiGRU:

```
Webcam frames → Backend (MediaPipe Holistic | MMPose)
             → (T, K, 4): x,y,z,confidence mỗi keypoint
             → _build_kinematic_features(): position + velocity + acceleration + confidence
             → (T, K×7)
             → BiGRU(input_dim, hidden=512, layers=2, bidirectional)
             → Linear(1024, 768)
             → (1, 1, 768)   ← cùng shape với ViT-ImageNet branch
```

**Tại sao đúng hơn ViT-ImageNet:**
- ViT mean-pool qua 16 frame → mù hoàn toàn với chuyển động (desk-slap, nhảy)
- Velocity + Acceleration bắt được impulse tức thời (đập bàn) và năng lượng tuần hoàn (nhảy)

**P0.2 — Confidence là tín hiệu, không phải rác:**
- Khi nhảy/chồm mạnh, keypoint confidence tụt → đây là chỉ báo arousal cao
- Confidence giữ nguyên trong feature vector, không filter bỏ

**Backend abstraction:**
- `_MediaPipeBackend`: chạy CPU, cài pip, 9 pose + 42 hand keypoints = 51 kpts
- `_MMPoseBackend`: chính xác hơn, cần CUDA, 13 COCO upper-body kpts
- Cùng output format `(T, K, 4)` → `PoseContextEncoder` không thay đổi khi đổi backend

---

## Phase 0 — Factory encoder

**File:** `src/vie_gameemo/encoders/__init__.py`

```python
from vie_gameemo.encoders import get_context_encoder

encoder = get_context_encoder(cfg)
# cfg.visual_encoder.context_encoder.type = "pose"        → PoseContextEncoder
# cfg.visual_encoder.context_encoder.type = "vit_imagenet" → ContextEncoder
```

---

## FIX 1 + FIX 2 — cue_extractor.py

**File:** `src/vie_gameemo/llm/cue_extractor.py`

### FIX 1 — Xóa context cue sai (brightness/color/edge)

**Vấn đề gốc:** Code cũ dùng OpenCV tính brightness, color variance, edge density trên **1 peak frame** → sai 4 mặt:
1. Thống kê ảnh tĩnh, không mô tả hành động
2. Tính trên 1 frame → không có chuyển động
3. Ánh sáng facecam streamer gần như cố định → signal gần bằng noise
4. Phá nhất quán feature–cue khi đổi sang pose encoder

**Sửa:** Xóa hoàn toàn `_precompute_visual_stats`, `_scene_cue`, tất cả brightness/edge bins.

**Thêm `MotionCueExtractor`** (chỉ cho nhánh pose):
- Precompute: chạy pose backend, tính motion energy, wrist impact, periodicity
- Extract: đọc cache JSON → discretize → named cue

```python
# Ví dụ output:
"motion — sharp downward arm impact"        # đập bàn: impact > 0.15
"motion — high whole-body energy, rhythmic" # nhảy/hype: energy cao + periodic
"motion — elevated body movement"           # chuyển động nhiều
"motion — minimal movement"                 # tĩnh
```

### FIX 2 — Attr vector gate theo branch

| Branch | Attr vector | n_attrs |
|--------|------------|---------|
| `pose` | face(5) + voice(3) + motion(3) + text(4) | **15** |
| `vit_imagenet` | face(5) + voice(3) + text(4) | **12** |

```python
ext = CueExtractor(context_encoder_type="vit_imagenet")
ext.n_attrs  # → 12

ext = CueExtractor(context_encoder_type="pose")
ext.n_attrs  # → 15
```

**`n_attrs` là dynamic property** — `GHeadPerModality` và training đều đọc từ đây.

---

## config.yaml — Thay đổi

```yaml
visual_encoder:
  context_encoder:
    type: "vit_imagenet"        # ← MỚI: "vit_imagenet" | "pose"
    # ... các key ViT-ImageNet cũ giữ nguyên ...
    
    # MỚI — pose branch:
    pose_backend: "mediapipe"   # "mediapipe" | "mmpose"
    pose_keypoints: "upper_body"
    pose_confidence_as_feature: true
    pose_temporal:
      type: "bigru"
      hidden_dim: 512
      n_layers: 2
      dropout: 0.1

training:
  llm1_explanation:
    g_head:
      hidden_dim: 128
      n_attrs_pose: 15           # ← MỚI (thay n_attrs: 15 cố định)
      n_attrs_vit_imagenet: 12   # ← MỚI

paths:
  checkpoint_vit_imagenet: null  # ← MỚI: checkpoint riêng mỗi nhánh
  checkpoint_pose: null          # ← MỚI
```

---

## Tests — Thay đổi

**File:** `tests/test_llm1_explanation.py`

| Test cũ | Hành động |
|---------|-----------|
| `TestGHead.test_forward_shape` | Xóa (GHead cũ không còn) |
| `TestCueExtractor.test_scene_cue_fallback` | Xóa (`_scene_cue` đã bị xóa) |
| `TestCueExtractor.test_n_attrs` | Tách thành `test_n_attrs_vit_imagenet` + `test_n_attrs_pose` |

**Tests mới thêm:**

| Test | Mô tả |
|------|-------|
| `TestGHeadPerModality::test_forward_shapes_with_context` | Shape với motion head |
| `TestGHeadPerModality::test_forward_shapes_no_context` | `motion_pred is None` khi vit_imagenet |
| `TestGHeadPerModality::test_heads_read_own_modality` | Mỗi head chỉ sensitive với input của mình |
| `TestCueExtractor::test_extract_vit_imagenet_no_context_cue` | Không có "motion" trong cue, attr 12-d |
| `TestCueExtractor::test_extract_pose_includes_motion_cue` | Có "motion" trong cue, attr 15-d |
| `TestMotionCueExtractor::test_extract_missing_cache` | `"no_motion_data"`, attrs=[0,0,0] |
| `TestMotionCueExtractor::test_extract_zero_kinematics` | Low energy → minimal cue |
| `TestMotionCueExtractor::test_extract_high_impact` | `"sharp downward arm impact"` |
| `TestMotionCueExtractor::test_extract_high_energy_rhythmic` | `"rhythmic"` |

---

## Bước tiếp theo (chưa làm — cần data + GPU)

### P0.3 — Retrain + Ablation
```bash
# Train nhánh ViT-ImageNet (baseline)
python scripts/train.py --config config.yaml \
  --set visual_encoder.context_encoder.type=vit_imagenet

# Train nhánh Pose
python scripts/train.py --config config.yaml \
  --set visual_encoder.context_encoder.type=pose \
  --set visual_encoder.context_encoder.pose_backend=mediapipe

# So sánh accuracy + đóng góp context qua harness ablation zero-context
```

### Precompute pose kinematics cache
```python
from vie_gameemo.llm.cue_extractor import CueExtractor
ext = CueExtractor(cache_dir="data/cache", context_encoder_type="pose")
ext.precompute_all(
    faces_dir="data/processed/faces",
    audios_dir="data/processed/audios",
    frames_dir="data/processed/contexts",
    pose_backend="mediapipe",
)
```

### Train LLM-1 (Stage A)
```bash
python scripts/train_llm1.py --config config.yaml \
  --set visual_encoder.context_encoder.type=pose
```
