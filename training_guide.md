# Hướng dẫn training: nhánh giải thích (LLM explanation) cho MLP cảm xúc

Tài liệu này giải thích *vì sao* thiết kế như vậy và *cách* train, dành cho setup: encoder 768-d × 4 modality → `ModalAdapter` (soft token) → LLM, song song với MLP phân loại `768→256→8`.

---

## 1. Ý tưởng cốt lõi

Bạn muốn LLM **giải thích vì sao MLP ra một nhãn cảm xúc**, nhưng chỉ có nhãn — không có lời giải thích viết tay. Hai năng lực cần tách bạch:

- **Phân loại (decide):** ánh xạ feature → nhãn. Học được từ loss phân loại.
- **Diễn đạt (verbalize):** nói ra bằng chữ *trung thực*. **Không** tự rơi ra từ việc học phân loại — cần gradient chảy qua phần sinh văn bản của LLM.

Vì thế "chỉ align soft token qua nhãn" sẽ cho ra classifier biết phân loại nhưng "giải thích" bằng cách **bịa** (confabulation). Giải pháp: cấp một tín hiệu mô tả/ngôn ngữ — nhưng lấy **tự động** từ detector đã có, nên gần như miễn phí.

### Hai điểm "tap"

| Tap | Lấy từ đâu | Vai trò | Tính chất |
|-----|-----------|---------|-----------|
| **B (neo)** | lớp áp chót MLP — vector **256-d sau GELU**, ngay trước `Linear(256,8)` | trung thực với quyết định | `nhãn = Linear(256,8)(B)` ⇒ chứa đúng quyết định; nhưng đã nén, nghèo chi tiết |
| **A (richness)** | raw token `audio/face/context/text` + `fusion` | chi tiết mô tả | giàu nội dung; nhưng không tự nó trung thực với MLP |

> `fusion` của bạn là conv-attention **trước** MLP, nên **không** phải tap B. Tap B là vector 256-d và phải được thêm vào `ModalAdapter` qua `proj_penult`.

Thứ tự soft token sau thay đổi: `[penult | fusion | audio | face | context | text]`.

## 2. Nhãn để học: distill từ MLP (không dùng gold)

Target trường `Emotion:` = `argmax` của **MLP**, kể cả khi MLP sai.

Vì sao không gold: mục tiêu là giải thích *quyết định của MLP*. Nếu MLP đoán "anger" mà gold là "joy", train trên gold sẽ bắt LLM giải thích "joy" — một quyết định MLP **không** đưa ra. Distill nhãn MLP còn nhất quán tuyệt đối với neo tap B (vì nhãn = hàm tuyến tính của penult).

Gold giữ lại **chỉ để đánh giá**: đo tỉ lệ MLP sai, và xem lời giải thích có "hedge" (giảm chắc chắn) đúng lúc không.

## 3. Auxiliary objective: generative-cue

Đổi target text thành:

```
Cues: {cues}. Emotion: {label}.
```

ví dụ:

```
Cues: voice — high pitch, fast rate, trembling; face — brow lowered (AU4),
lip tightener (AU23); text — negative-valence lexicon. Emotion: anger.
```

Train bằng **LM loss thường**. Để sinh được phần `Cues`, LLM **buộc** phải đọc tap A (chi tiết mịn không nằm trong tap B đã nén). Lợi ích kép: (1) tạo áp lực ngôn ngữ lên tap A → soft token rơi vào vùng *verbalize được*, không chỉ vùng *phân biệt lớp*; (2) phần `Cues` chính là **bộ khung lời giải thích**.

**Quy tắc bắt buộc — cue phải trực giao nhãn.** Dùng thuộc tính mịn (pitch/energy/rate, AU, lexicon). **Cấm** nhét cảm xúc vào cue ("sounds angry"): nếu cue suy được từ nhãn thì tap B đã thỏa được, và tap A lại bị bỏ qua — đúng failure mode ta đang chống.

Nguồn cue: từ detector đã có (prosody / openSMILE, Action Unit / OpenFace, lexicon). `CueExtractor` map output detector → chuỗi cue theo template.

## 4. Hàm loss

```
L = L_LM(cues + emotion | soft_tokens)                    # chính
  + λ_kl * KL(LLM_label_dist || softmax(MLP_logits / T))  # tuỳ chọn
  + λ_rec * smooth_l1(g(z_raw), attr_vector)              # tuỳ chọn (guard)
```

- `L_LM`: cross-entropy token-level trên cả `Cues` lẫn `Emotion`.
- `λ_kl` (vd 0.1): cho LLM thừa hưởng **độ tự tin** của MLP → giải thích các ca low-confidence trung thực hơn.
- `λ_rec` (vd 0.1): head nhỏ `g` **chỉ** đọc raw token (tap A) → tái tạo vector thuộc tính số. Chặn vật lý đường tắt "bỏ qua tap A". Bật khi ablation cho thấy Cues không phụ thuộc tap A.

## 5. Quy trình train 2 stage

| | Stage A — alignment | Stage B — instruction-tune nhẹ |
|---|---|---|
| Train | `ModalAdapter` (+ `proj_penult`, head `g`) | thêm LoRA trên LLM |
| Freeze | LLM, MLP, encoders | MLP, encoders |
| LR | cao hơn (vd 1e-4) | thấp (vd 2e-5 cho LoRA) |
| Mục đích | đưa soft token vào vùng LLM đọc được | tinh chỉnh văn phong/độ bám |

Đóng băng tuyệt đối MLP + encoders (`requires_grad=False`, `eval()`, `no_grad` khi lấy penult & nhãn). MLP là hệ thống cần giải thích — đổi nó là đổi luôn thứ cần giải thích.

### Siêu tham số gợi ý (điểm khởi đầu)

```
d_llm            : theo model bạn dùng
lora_rank        : 4–8       (dataset nhỏ → giữ thấp)
lora_alpha       : 16
weight_decay     : 0.01–0.05
dropout (adapter): 0.1–0.3
batch size       : theo VRAM
T (distill)      : 2.0
λ_kl, λ_rec      : 0.1
modality dropout : 0.1–0.2   (augmentation, tận dụng zero-mask)
```

## 6. Dataset nhỏ (3399 samples, 8 lớp) — chống overfitting

- **Ưu tiên Stage A.** Vào Stage B chỉ với LoRA rank thấp + weight decay + early stopping theo **metric eval faithfulness**, không theo train loss. Nếu val xấu đi, dừng ở Stage A.
- **Modality dropout làm augmentation:** random drop 1 modality (dùng đúng zero-mask sẵn có) mỗi step.
- **Generative cue = nhiều bit/sample hơn nhãn trần** ⇒ supervision dày hơn, hợp data nhỏ. Đây là lợi thế, không phải chi phí.
- Theo dõi gap train/val sớm; LLM lớn rất dễ nhớ 3399 mẫu.

## 7. Đánh giá faithfulness (đừng tin loss đẹp)

1. **Ablation tap A:** zero/nhiễu raw token lúc inference. Kỳ vọng: phần `Cues` **suy giảm rõ**, còn `Emotion` (từ tap B) **gần như giữ nguyên**. Nếu Cues không đổi ⇒ model bỏ qua tap A → tăng `λ_rec` hoặc chọn cue trực giao hơn.
2. **Agreement nhãn:** `argmax` LLM vs `argmax` MLP. Phải **cao** (cùng đọc penult). Thấp ⇒ neo tap B chưa được dùng.
3. **Counterfactual:** bóp một modality đầu vào → MLP đổi nhãn; kiểm lời giải thích có đổi *nhất quán* theo hướng đó không. Nhãn đổi mà giải thích đứng im = đang biện minh.
4. **NN-decode (sanity):** chiếu soft token về token vocab gần nhất; toàn rác ⇒ activation vector chứ không phải biểu diễn ngữ nghĩa.
5. **Hedge khi MLP sai:** trên tập gold, xem các ca MLP sai lời giải thích có giảm chắc chắn không.

## 8. Phân biệt nhanh hai loại "trung thực"

- **Trung thực về thông tin** (tap B đảm bảo): cái LLM đọc đúng là cái MLP dùng để quyết định.
- **Trung thực của lời giải thích** (generative-cue + ablation đảm bảo): chữ nói ra phản ánh tín hiệu thật, không bịa.

Cần **cả hai**. tap B lo cái thứ nhất; cue + guard + ablation lo cái thứ hai.

## 9. Thứ tự triển khai gợi ý

1. Lộ `penult` từ MLP → thêm `proj_penult`, cập nhật mask & thứ tự token.
2. `CueExtractor` + pipeline target `"Cues: … Emotion: …"` (label = MLP argmax).
3. Stage A, kiểm bằng ablation tap A + agreement.
4. Bật `λ_rec`/`λ_kl` nếu ablation chỉ ra cần.
5. Stage B (LoRA) — chỉ khi Stage A ổn và val không overfit.
