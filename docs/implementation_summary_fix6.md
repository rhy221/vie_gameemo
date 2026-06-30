# Implementation Summary — LLM-1 FIX 6.1–6.6

**Ngày:** 2026-06-30  
**Repo:** `vie-gameemo-skeleton`  
**Kết quả test:** 113/113 passed, 0 regression

---

## Tổng quan

Thực hiện theo `claude_code_prompt (2).md` — 6 correctness fix trên code đã implement ở phiên trước (Phase 0 + FIX 1–5).

---

## FIX 6.3 + 6.5 — Kinematic masking + dimension comment

**File:** `src/vie_gameemo/encoders/context_pose.py` — hàm `_build_kinematic_features()`

### FIX 6.3 — Confidence-masked derivatives

**Vấn đề gốc:** Velocity và acceleration được tính vô điều kiện. Khi một keypoint bị mất (confidence thấp) rồi xuất hiện lại, tạo ra spike velocity giả lớn — gây nhiễu signal thực sự.

**Sửa:** Nhân derivative với tích confidence của các frame liền kề:

```python
vel_mask = conf[1:] * conf[:-1]              # (T-1, K, 1)
vel[1:] = raw_vel * vel_mask                 # zero khi endpoint không chắc

acc_mask = conf[2:] * conf[1:-1] * conf[:-2] # (T-2, K, 1)
acc[2:] = raw_acc * acc_mask
```

**Giữ nguyên confidence trong feature vector** (không filter bỏ) — confidence thấp vẫn là tín hiệu arousal cao (keypoint mất khi nhảy mạnh, đập bàn).

### FIX 6.5 — Clarification comment

Thêm ghi chú trong docstring:

```
2D only — z intentionally dropped; 7 dims = pos(2) + vel(2) + acc(2) + conf(1)
```

---

## FIX 6.2 — Giảm kích thước BiGRU

**Files:** `src/vie_gameemo/encoders/context_pose.py`, `src/vie_gameemo/encoders/__init__.py`, `config.yaml`

**Vấn đề:** `hidden_dim=512, n_layers=2` — quá nhiều tham số cho 3399 sample → nguy cơ overfit.

| Tham số | Cũ | Mới |
|---|---|---|
| `hidden_dim` | 512 | 128 |
| `n_layers` | 2 | 1 |
| `dropout` | 0.1 | 0.3 |

**Tại sao 128/1:** BiGRU(128) bidirectional → 256-d hidden → Linear(256, 768). Input đã giàu (357-d cho MediaPipe, 91-d cho MMPose); GRU chỉ cần aggregate qua thời gian, không cần học hierarchy sâu.

Cập nhật tại 3 chỗ: default args của `PoseContextEncoder.__init__`, fallback defaults của factory trong `encoders/__init__.py`, và `pose_temporal` trong `config.yaml`.

---

## FIX 6.4 — Comment xác nhận KL direction

**File:** `src/vie_gameemo/training/llm1_explanation.py` — block `L_kl` (line 593)

Không thay đổi code, chỉ thêm 2 dòng comment trước `if lambda_kl > 0`:

```python
# KL(mlp_teacher || llm_student): lm_soft=log_softmax(llm/T), mlp_soft=softmax(mlp/T)
# F.kl_div(log_Q, P) = KL(P||Q) — direction: MLP→LLM ✓; T² restores gradient magnitude
```

Giải thích rõ:
- `F.kl_div(log_Q, P)` = `KL(P||Q)` → teacher = MLP (P), student = LLM (Q) ✓
- Nhân `T²` để khôi phục magnitude của gradient bị mất khi chia logits cho T

---

## FIX 6.6 — Chuyển hardcoded thresholds sang config

**Files:** `config.yaml`, `src/vie_gameemo/llm/cue_extractor.py`, `src/vie_gameemo/evaluation/faithfulness.py`

### Thresholds đã được chuyển

| Giá trị cũ (hardcoded) | Vị trí | Config key |
|---|---|---|
| `_MOTION_ENERGY_BINS = [(0.02,"low"),(0.08,"moderate"),...]` | `cue_extractor.py:51` | `cues.motion_energy_bins` |
| `_IMPACT_BINS = [(0.05,"none"),(0.15,"moderate"),...]` | `cue_extractor.py:52` | `cues.impact_bins` |
| `_PERIOD_BINS = [(0.2,"none"),(0.5,"weak"),...]` | `cue_extractor.py:53` | `cues.periodicity_bins` |
| `top_prob < 0.6` | `faithfulness.py` | `evaluation.faithfulness.hedge_threshold` |

### Config additions (`config.yaml`)

```yaml
cues:
  motion_energy_bins: [[0.02, "low"], [0.08, "moderate"]]
  impact_bins:        [[0.05, "none"], [0.15, "moderate"]]
  periodicity_bins:   [[0.20, "none"], [0.50, "weak"]]

evaluation:
  faithfulness:
    hedge_threshold: 0.6
    n_samples: 200
```

### Code changes (backward-compatible)

**`CueExtractor.__init__`** — thêm param `cues_cfg=None`:
- Khi có config: parse bins từ config và truyền xuống `MotionCueExtractor`
- Khi `None`: dùng module-level constants (backward compat)

**`MotionCueExtractor.__init__`** — thêm optional `motion_energy_bins`, `impact_bins`, `period_bins`:
- Lưu dưới `self._motion_energy_bins` v.v.
- `extract()` dùng instance bins thay vì constants

**`eval_hedge()`** — thêm param `hedge_threshold: float = 0.6` (configurable, không còn hardcoded).

---

## FIX 6.1 — Rewrite `eval_hedge()`: correlation metric

**File:** `src/vie_gameemo/evaluation/faithfulness.py`

### Vấn đề cũ

`eval_hedge()` cũ chỉ chạy trên các sample MLP sai (`mlp_idx != gt_idx`) rồi đo xem LLM có tự tin thấp không. Metric phụ thuộc vào gold label → đây là **calibration metric**, không phải **faithfulness metric**.

### Logic mới

**Primary metric:** Pearson + Spearman correlation giữa MLP confidence và LLM Emotion-token confidence trên toàn bộ sample.

```
Với mỗi sample:
  mlp_conf = max(softmax(logits))          # độ tự tin của MLP
  llm_conf = prob(emotion_label_token)     # xác suất token label tại generation step

→ pearsonr(mlp_confs, llm_confs)
→ spearmanr(mlp_confs, llm_confs)
```

Một explainer faithful sẽ có correlation dương: tự tin cao khi MLP tự tin, giảm khi MLP bất định.

**Secondary metric (giữ lại):** `hedge_rate = n_hedged / n_mlp_wrong` — khi MLP sai, LLM có giảm confidence xuống dưới `hedge_threshold` không.

### Return keys mới

```python
{
    "hedge_confidence_pearson_r": float,     # PRIMARY
    "hedge_confidence_pearson_p": float,
    "hedge_confidence_spearman_r": float,    # PRIMARY
    "hedge_confidence_spearman_p": float,
    "hedge_n_mlp_wrong": int,                # secondary
    "hedge_n_hedged": int,                   # secondary
    "hedge_rate": float,                     # secondary
    "hedge_n_samples": int,
}
```

### Implementation notes

- Chạy trên **tất cả** sample (không gating `mlp_wrong`), collect list `mlp_confs`, `llm_confs`
- `scipy.stats.pearsonr/spearmanr` — degrade gracefully nếu scipy chưa cài (log warning, return 0.0)
- `hedge_threshold` không còn hardcoded, đọc từ param (default 0.6)

---

## Tóm tắt file đã thay đổi

| File | Thay đổi |
|------|---------|
| `src/vie_gameemo/encoders/context_pose.py` | FIX 6.2 (BiGRU defaults), FIX 6.3 (vel/acc masking), FIX 6.5 (comment) |
| `src/vie_gameemo/encoders/__init__.py` | FIX 6.2 (factory fallback defaults) |
| `src/vie_gameemo/training/llm1_explanation.py` | FIX 6.4 (KL comment) |
| `src/vie_gameemo/llm/cue_extractor.py` | FIX 6.6 (cues_cfg param, instance bins) |
| `src/vie_gameemo/evaluation/faithfulness.py` | FIX 6.1 (correlation metric), FIX 6.6 (hedge_threshold param) |
| `config.yaml` | FIX 6.2 (pose_temporal sizes), FIX 6.6 (cues + evaluation.faithfulness sections) |
