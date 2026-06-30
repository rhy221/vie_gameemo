# Prompt cho Claude Code — thêm nhánh giải thích (explanation) bằng LLM

> Paste toàn bộ nội dung dưới đây vào Claude Code khi đang mở repo `vie-gameemo-skeleton`.

---

## Bối cảnh

Project hiện có:
- Các encoder (audio / face / context / text) cho ra feature 768-d mỗi modality.
- Một `ModalAdapter` với 5 `nn.Linear` riêng (`proj_fusion, proj_audio, proj_face, proj_context, proj_text`, tất cả `768 → d_llm`), concat thành `[fusion | audio | face | context | text]`, dùng zero attention mask cho modality bị thiếu. `fusion` đến từ một module **conv-attention** fuse 4 modality.
- Một MLP phân loại cảm xúc: `mean-pool → Linear(768,256) → GELU → Dropout(0.3) → Linear(256,8)` (8 lớp, ~3399 samples). File: `src/vie_gameemo/classifiers/mlp.py`.
- Một LLM nhận soft token từ `ModalAdapter`.

**Mục tiêu:** LLM sinh ra lời giải thích *trung thực với quyết định của MLP* và *giàu chi tiết mô tả*, mà ta **chỉ có nhãn cảm xúc** (không có rationale viết tay). Đã có sẵn detector mô tả (prosody / Action Unit / lexicon).

## Nguyên tắc thiết kế (đọc kỹ — quyết định mọi thứ)

1. **Hai điểm "tap":**
   - **tap B = neo faithfulness** = lớp áp chót của MLP, tức **vector 256-d sau GELU**, ngay trước `Linear(256,8)`. Vì `logits = Linear(256,8)(penult)`, vector này chứa *đúng* quyết định của MLP. **Hiện chưa được đưa vào LLM — phải thêm.**
   - **tap A = nguồn richness** = các raw modality token (`audio/face/context/text`) đã có, cộng `fusion`. Chúng giữ chi tiết mịn mà bottleneck phân loại của MLP đã nén bỏ.
2. **Distill nhãn từ MLP, KHÔNG dùng gold.** Target trường `Emotion:` = `argmax` của MLP (kể cả khi MLP sai). Gold chỉ dùng để eval.
3. **Auxiliary objective kiểu generative-cue:** đổi target text của LLM thành
   `"Cues: {cues}. Emotion: {label}."`
   rồi train bằng LM loss thường. Phần `{cues}` (chi tiết mô tả per-modality từ detector) **buộc** LLM phải đọc tap A để sinh ra được — vì các cue đó không nằm trong tap B đã nén. Đây là cách "chế" supervision mô tả tự động từ thứ ta đang có.
4. **Cue phải trực giao với nhãn.** Dùng thuộc tính mịn (pitch/energy/rate, AU, lexicon hit), **không** nhét chính cảm xúc vào cue ("sounds angry" = rò nhãn → cấm).
5. **Đóng băng MLP + encoders.** MLP là "hệ thống cần được giải thích" — không được thay đổi (chạy `eval()` + `no_grad`). Chỉ `ModalAdapter` (+ LoRA ở stage 2) được train.

## Việc cần làm (theo thứ tự)

### B0. Khảo sát repo trước khi sửa
Đọc và tóm tắt cho tôi: `src/vie_gameemo/classifiers/mlp.py`, định nghĩa `ModalAdapter`, vòng training hiện tại, cách load LLM (model nào, `d_llm`, frozen hay không), định dạng một sample dataset (feature nằm ở đâu, shape ra sao), và vị trí các detector mô tả đã có. **Chưa sửa gì** — báo cáo hiểu biết + kế hoạch, rồi mới code.

### B1. Lộ ra lớp áp chót của MLP
Trong `mlp.py`, refactor `forward` để có thể trả về **cả** `logits` và `penult` (vector 256-d *input của `Linear(256,8)`*). Cách sạch nhất: tách head cuối ra, hoặc dùng forward hook. Phải lấy penult ở chế độ `eval()` (dropout = identity) để ổn định.

### B2. Thêm projection neo vào ModalAdapter
- Thêm `proj_penult = nn.Linear(256, d_llm)`.
- Thứ tự soft token mới: `[penult | fusion | audio | face | context | text]`.
- Cập nhật attention mask: `penult` và `fusion` luôn present; raw token theo availability như cũ.
- Giữ nguyên convention số token / chuẩn hoá hiện có; match shape thực tế trong code.

### B3. Pipeline tạo target text (cue + label)
- Viết `CueExtractor`: nhận output các detector đã có → trả về chuỗi cue *trực giao nhãn*, gọn, theo template, ví dụ:
  `voice: high pitch, fast rate, trembling; face: brow lowered (AU4), lip tightener (AU23); text: negative-valence lexicon`
- Lấy `mlp_pred = argmax(MLP(features))` (MLP frozen, no_grad).
- Target text: `f"Cues: {cue_str}. Emotion: {LABELS[mlp_pred]}."`
- Lưu thêm `gold_label` vào batch để eval (không vào loss).

### B4. Hàm loss
```
L = L_LM(target_text | soft_tokens)                      # chính
  [+ λ_kl * KL(LLM_label_dist || softmax(MLP_logits/T))] # tuỳ chọn: soft-distill độ tự tin của MLP
  [+ λ_rec * smooth_l1(g(z_raw), attr_vector)]           # tuỳ chọn: guard retention cho tap A
```
- `L_LM`: cross-entropy token-level trên `target_text` (cả phần Cues lẫn Emotion).
- `g`: MLP head nhỏ, **chỉ** đọc mean-pool của raw modality token (`z_raw`, tap A) → tái tạo vector thuộc tính số từ detector. Mục đích: chặn đường tắt "bỏ qua tap A". Đặt `λ_rec` nhỏ (vd 0.1), `λ_kl` nhỏ (vd 0.1); để bật/tắt qua config.

### B5. Script train 2 stage
- **Stage A (alignment):** freeze LLM + MLP + encoders; chỉ train `ModalAdapter` (gồm `proj_penult`) + head `g`. Loss như trên.
- **Stage B (instruction-tune nhẹ):** thêm LoRA lên LLM (rank thấp 4–8), giữ MLP + encoders frozen, tiếp tục cùng loss.
- Cho phép chạy **chỉ Stage A** qua flag (vì dataset nhỏ, Stage B có thể overfit).
- Config hoá: `λ_kl, λ_rec, lora_rank, lr theo stage, dropout, freeze flags, run_stage_b`.

### B6. Đánh giá faithfulness (bắt buộc thêm)
Viết script eval:
- **Ablation tap A:** zero/nhiễu các raw token lúc inference → kỳ vọng phần `Cues` suy giảm rõ trong khi `Emotion` (từ tap B) gần như giữ nguyên. Nếu Cues *không đổi* ⇒ model đang bỏ qua tap A (tăng `λ_rec` hoặc chọn cue trực giao hơn).
- **Agreement:** so `argmax` nhãn LLM sinh vs `argmax` MLP. Mục tiêu cao (vì cùng đọc penult). Bất đồng nhiều ⇒ neo tap B chưa được dùng đúng.
- **NN-decode (sanity):** chiếu mỗi soft token về token vocab gần nhất; ghi log để kiểm tra chúng có rơi vào vùng ngữ nghĩa hay chỉ là activation vector.
- Báo cả accuracy nhãn-LLM so với **gold** để biết model có "hedge" khi MLP sai không.

## Ràng buộc & tiêu chí nghiệm thu
- MLP + encoders không bao giờ nhận gradient (assert no_grad / `requires_grad=False`).
- Không thay đổi hành vi phân loại của MLP gốc.
- Code chạy được end-to-end trên một batch nhỏ (viết 1 smoke test).
- Mọi siêu tham số mới nằm trong file config, có default hợp lý.
- Trước khi code, trình bày plan + chỗ định sửa; sau khi code, chạy smoke test và tóm tắt thay đổi.

## Lưu ý dataset nhỏ (3399 samples, 8 lớp)
- Ưu tiên Stage A; vào Stage B chỉ với LoRA rank thấp + weight decay + early stopping theo **metric eval faithfulness**, không theo train loss.
- Tận dụng zero-mask modality như **augmentation** (random drop 1 modality khi train).
- Generative cue cho nhiều bit/sample hơn nhãn trần ⇒ tốt cho data nhỏ; tận dụng điều này.
