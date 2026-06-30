# Implementation Summary — LLM-1 FIX 8.1–8.4

**Ngày:** 2026-07-01  
**Repo:** `vie-gameemo-skeleton`  
**Kết quả test:** 118/118 passed (116 cũ + 2 test mới từ FIX 8.1 + FIX 8.3), 0 regression

---

## Tổng quan

Thực hiện theo `claude_code_prompt (4).md` — rà soát FIX 7, sửa 4 điểm còn hở.

---

## FIX 8.1 — Xác nhận + test cho 3-frame window gate

**File:** `src/vie_gameemo/llm/cue_extractor.py`, `tests/test_llm1_explanation.py`

**Vấn đề mô tả:** FIX 7.1 "nhiều khả năng gate sai khung" — nếu binary gate chỉ kiểm `conf[t]` (khung impact), sẽ không chặn được spike giả khi cổ tay mất tracking ở frame t-1, t-2 rồi xuất hiện lại với conf cao ở frame t.

**Kết quả kiểm tra:** Implementation từ FIX 7.1 ĐÃ window-aware:
```python
min_conf_3frame[2:] = np.minimum(
    np.minimum(wrist_conf[2:], wrist_conf[1:-1]), wrist_conf[:-2]
)
```
Gate kiểm `min(conf[t], conf[t-1], conf[t-2])` — khớp đúng 3 frame cần để tính `acc[t]`.

**Thêm test bắt đúng bug mà data sạch giấu:**
```
TestMotionCueWindowGate.test_gap_tracking_does_not_produce_false_impact
```
- Tạo kps với gap (frames 5-9 conf=0.05, pos=999.0)
- Frame 10 reappears với conf=0.95 → `velocity[10] = 999 - garbage = spike`
- Không có gate: impact ≈ 1998 (fake)
- Có gate: `min(conf[10], conf[9], conf[8]) = 0.05 < 0.3` → gate=0 → impact=0 ✓

---

## FIX 8.2 — Tài liệu hoá sự khác biệt có chủ ý giữa soft mask và binary gate

**File:** `src/vie_gameemo/llm/cue_extractor.py`

Thêm comment giải thích tại sao feature (context_pose.py) dùng soft mask còn cue (cue_extractor.py) dùng binary gate:
- Feature: continuous confidence weighting — att chỉ làm giảm derivative, không loại hẳn
- Cue: hard 0/1 gate — impact label phải "có/không", không phải "0.06× thật"
- Cả hai đều kiểm cùng 3-frame window

---

## FIX 8.3 — Multi-token label: warning → hard ValueError

**File:** `src/vie_gameemo/training/llm1_explanation.py`

**Vấn đề:** `_build_label_token_ids()` cảnh báo khi nhãn multi-token rồi vẫn ngầm lấy token đầu → nhập nhằng còn nguyên.

**Sửa:** Đổi `logger.warning(...)` thành `raise ValueError(...)`:
```python
if len(toks) > 1:
    raise ValueError(
        f"Label '{name}' encodes as {len(toks)} tokens {toks}; "
        "all emotion labels must map to a single token for consistent "
        "KL and hedge metrics. Rename the label to a single-token surface form."
    )
```

Áp dụng nhất quán cho cả KL (`_extract_emotion_logprobs`) lẫn hedge (`eval_hedge` trong faithfulness.py qua import).

**Test mới:**
```
TestKLLossFormula.test_label_multi_token_raises
```
Dùng MockTokenizer trả [10, 11] cho nhãn "b" → raises ValueError với match "single token|single-token".

---

## FIX 8.4 — Xác nhận KL restrict 8 token + renormalize

**File:** `src/vie_gameemo/training/llm1_explanation.py`

**Kiểm tra:** `_extract_emotion_logprobs()` trả `lm_logits[i, last_pos, label_token_ids]` → shape `(B, 8)` raw logits cho đúng 8 token. Sau đó `F.log_softmax(…, dim=-1)` normalize TRÊN 8 token → sum-to-1 trên 8-token support, khớp MLP distribution.

**Thêm:**
1. Assert shape trong `_extract_emotion_logprobs`:
   ```python
   assert token_logits.shape == (n_labels,)
   result = torch.stack(emotion_logits)
   assert result.shape == (B, n_labels)
   ```
2. Comment xác nhận trong cả `_extract_emotion_logprobs` và `_compute_loss`:
   ```python
   # Both distributions are 8-d (same support): mlp_soft=(B,8), lm_soft=log_softmax over 8 tokens (FIX 8.4)
   lm_soft = F.log_softmax(lm_logits_at_emotion / kl_temp, dim=-1)  # log-normalized over 8 tokens ✓
   ```

---

## Tóm tắt file đã thay đổi

| File | Thay đổi |
|------|---------|
| `src/vie_gameemo/llm/cue_extractor.py` | FIX 8.2 (comment tài liệu hoá soft vs binary mask) |
| `src/vie_gameemo/training/llm1_explanation.py` | FIX 8.3 (ValueError cho multi-token), FIX 8.4 (shape assert + comment) |
| `tests/test_llm1_explanation.py` | FIX 8.1 (TestMotionCueWindowGate gap test), FIX 8.3 (test_label_multi_token_raises) |
