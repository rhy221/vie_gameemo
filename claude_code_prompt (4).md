# Prompt cho Claude Code — context encoder (pose) + nhánh giải thích LLM

> Paste toàn bộ vào Claude Code khi đang mở repo `vie-gameemo-skeleton`.

---

## Bối cảnh dự án

Phân loại cảm xúc multimodal cho **streamer**, 4 modality:
- **face** — crop mặt từ facecam → ViT-FER → 768-d.
- **audio** — Whisper → 768-d.
- **transcript** — text encoder → 768-d.
- **context** — facecam thấy **cả thân trên**, mô tả *streamer đang làm gì* (đập bàn = giận; nhảy nhót = hype). **Hiện dùng ViT-ImageNet — đây là lựa chọn sai, sẽ thay (xem Phase 0).**

4 modality → conv-attention **fusion** → `ModalAdapter` (5 `nn.Linear 768→d_llm`: `proj_fusion/audio/face/context/text`, concat `[fusion|audio|face|context|text]`, zero-mask cho modality thiếu) → soft token → LLM. Song song: MLP phân loại `mean-pool → Linear(768,256) → GELU → Dropout(0.3) → Linear(256,8)` (8 lớp, ~3399 samples). File MLP: `src/vie_gameemo/classifiers/mlp.py`.

**Mục tiêu cuối:** LLM sinh lời giải thích *trung thực với quyết định MLP* và *giàu chi tiết mô tả*, trong khi ta chỉ có nhãn cảm xúc (không có rationale viết tay).

## Nguyên tắc thiết kế (đọc kỹ — chi phối toàn bộ)

1. **Hai "tap":**
   - **tap B = neo faithfulness** = lớp áp chót MLP, **vector 256-d sau GELU**, input của `Linear(256,8)`. Vì `logits = Linear(256,8)(penult)`, nó chứa *đúng* quyết định. **Hiện chưa đưa vào LLM — phải thêm.**
   - **tap A = richness** = raw token `face/audio/transcript/context` + `fusion`. Giữ chi tiết mịn mà tap B đã nén bỏ.
   - Thứ tự token mới: `[penult | fusion | face | audio | transcript | context]`.
2. **Distill nhãn từ MLP, KHÔNG dùng gold.** Trường `Emotion:` = `argmax` MLP (kể cả khi sai). Gold chỉ để eval.
3. **Generative-cue objective:** target text = `"Cues: {cues}. Emotion: {label}."`, train bằng LM loss. Phần `{cues}` buộc LLM đọc tap A; nó cũng là bộ khung lời giải thích.
4. **Cue phải TRỰC GIAO NHÃN.** Dùng thuộc tính mịn mức thấp, **không** dùng thứ suy thẳng ra cảm xúc:
   - **face:** Action Unit (AU4, AU12, AU23…) — **KHÔNG** dùng output cảm xúc của ViT-FER (rò nhãn) và **KHÔNG** dùng embedding ViT-FER (không đọc được). Cần detector AU riêng.
   - **audio:** prosody thô (pitch/energy/rate/jitter) — **KHÔNG** dùng model SER (rò nhãn). Whisper embedding không cho prosody đọc được → cần extractor riêng. (Riêng *tốc độ nói* có thể suy từ Whisper word-timestamps.)
   - **transcript:** trừu xuất (valence/intensity/negation/profanity) — **KHÔNG** copy nguyên câu vào cue (rò nhãn qua từ vựng).
   - **context:** kinematic (motion energy/velocity/impact) — **KHÔNG** dùng action-recognition trả nhãn hành động (rò nhãn).
5. **NHẤT QUÁN feature–cue (theo từng nhánh encoder):** chỉ sinh cue cho thông tin mà *feature gốc thật sự mang*. Context có **hai nhánh chọn được qua config** (xem Phase 0): `pose` và `vit_imagenet` (nhánh ablation). Cue context phải **khớp nhánh đang bật**: `pose` → motion/action cue; `vit_imagenet` → **không** sinh action cue (feature này mù hành động) — để trống context cue cho nhánh đó. Tuyệt đối không bịa action cue khi nhánh active là `vit_imagenet`.
6. **Đóng băng tuyệt đối MLP + tất cả encoders** trong Phase 1 (`requires_grad=False`, `eval()`, `no_grad` khi lấy penult/nhãn). Chỉ `ModalAdapter` (+ `proj_penult`) và LoRA-LLM được train.

---

# PHASE 0 — Thêm context encoder pose-kinematics, GIỮ ViT-ImageNet làm nhánh ablation

> Không xoá ViT-ImageNet. Biến context encoder thành **thành phần chọn được qua config** với (ít nhất) hai nhánh: `vit_imagenet` (hiện tại, baseline ablation) và `pose` (mới). Mỗi nhánh cho feature khác nhau ⇒ **train lại fusion + MLP riêng cho từng nhánh**, rồi so sánh bằng harness ablation context (chủ dự án đã có) + metric phân loại. Để dữ liệu quyết định pose có thật sự thắng ViT-ImageNet không, thay vì giả định.

### P0.0 — Khảo sát
Đọc & báo cáo: context encoder hiện tại (ViT-ImageNet) được gọi ở đâu, nhận **một khung hay chuỗi khung**, ghép theo thời gian thế nào (nghi vấn: mean-pool → mù chuyển động). Đề xuất chỗ chèn điểm chọn nhánh (config `context_encoder: vit_imagenet | pose`). Báo plan trước khi sửa.

### P0.1 — Thêm nhánh pose (không thay thế)
- Refactor để context encoder chọn được qua config; giữ nguyên đường `vit_imagenet` chạy được như cũ.
- Thêm nhánh `pose` = **MediaPipe Holistic** (hoặc MMPose): keypoint **thân trên + hai tay + đầu** theo từng khung → chuỗi keypoint.
- Trên chuỗi keypoint, thêm **module thời gian** (TCN / GRU nhỏ / ST-GCN) ra feature 768-d (khớp chiều để fusion dùng được cho cả hai nhánh). **Tín hiệu phải tính trên CHUYỂN ĐỘNG giữa các khung** (vận tốc, gia tốc, jerk, motion energy) — KHÔNG mean-pool pose tĩnh.
- Phủ hai chế độ: **xung cục bộ** (đập bàn = gia tốc tay/thân hướng xuống nhọn) và **năng lượng toàn thân tuần hoàn** (nhảy/hype).

### P0.2 — Keypoint thiếu/độ tin cậy thấp = TÍN HIỆU, không phải rác
Khi nhảy/chồm mạnh, keypoint dễ mất hoặc confidence tụt. KHÔNG vứt mẫu. Dùng confidence như feature, và coi **độ biến động/độ tin cậy tụt mạnh cũng là chỉ báo arousal cao**. (Đúng các khoảnh khắc hype đỉnh điểm — đừng để pipeline ngây thơ làm sạch mất chúng.)

### P0.3 — Retrain mỗi nhánh + so sánh (ablation)
Train lại conv-attention fusion + MLP cho **từng nhánh** (`vit_imagenet`, `pose`), lưu checkpoint riêng theo nhánh. Báo: accuracy mỗi nhánh, và đóng góp của modality context trong mỗi nhánh qua harness ablation đã có (zero context → đo sụt). Mục tiêu là biết pose có cải thiện thật so với ViT-ImageNet baseline không. Config phải trỏ đúng `fusion/MLP checkpoint` theo `context_encoder` đang chọn.

---

# PHASE 1 — Nhánh giải thích LLM

### P1.1 — Lộ lớp áp chót MLP (tương thích ngược)
Trong `mlp.py`, sửa `forward(..., return_penult: bool = False)`: mặc định trả `logits` như cũ; khi `return_penult=True` trả `(logits, penult)` với `penult` = vector 256-d *input của `Linear(256,8)`*. **Không** đổi layer/trọng số/luồng tính; chỉ lộ activation (forward hook hoặc tách head). Đảm bảo không phá call site hiện có.

### P1.2 — proj_penult vào ModalAdapter
Thêm `proj_penult = nn.Linear(256, d_llm)`. Token order: `[penult | fusion | face | audio | transcript | context]`. `penult` & `fusion` luôn present; raw token theo availability (zero-mask như cũ). Giữ convention số token/chuẩn hoá hiện có.

### P1.3 — Cue extractors (một class mỗi modality, trực giao nhãn)
Mỗi extractor: detector → số thô → **rời rạc hóa bằng ngưỡng** (ngưỡng theo phân vị tập train, vd top/bottom ~30%) → chuỗi cue ngắn theo template. Giữ vài cue *nổi bật* mỗi modality, không liệt kê tất cả.
- `FaceCueExtractor`: AU từ **OpenFace 2.0 / py-feat / MediaPipe blendshapes** → `face — brow lowered (AU4), lip tightener (AU23)`.
- `AudioCueExtractor`: **librosa / openSMILE(eGeMAPS) / parselmouth** → pitch (pyin), RMS energy, speaking rate, jitter → `voice — high pitch, fast rate, trembling`.
- `TextCueExtractor`: lexicon (NRC-VAD / sentiment tagger có sẵn) → `text — negative valence, profanity, exclamatory` (KHÔNG copy câu).
- `MotionCueExtractor` **(chỉ khi Phase 0 đã làm)**: từ kinematics → `motion — sharp downward arm impact` (đập bàn) / `motion — high whole-body energy, rhythmic` (nhảy).
- Một `CueAssembler` gộp các modality present thành chuỗi `Cues:` cuối; bỏ qua modality thiếu.

### P1.4 — Target text
`mlp_pred = argmax(MLP(features, return_penult=True))` (frozen, no_grad). Target = `f"Cues: {cue_str}. Emotion: {LABELS[mlp_pred]}."`. Lưu thêm `gold_label` vào batch (chỉ eval, không vào loss).

### P1.5 — Loss
```
L = L_LM(cues + emotion | soft_tokens)                       # chính
  [+ λ_kl * KL(LLM_label_dist || softmax(MLP_logits / T))]   # tuỳ chọn: distill độ tự tin MLP
  [+ λ_rec * smooth_l1(g(z_rawA), attr_vector)]              # tuỳ chọn: guard retention tap A
```
`g` = MLP head nhỏ **chỉ** đọc mean-pool raw modality token (tap A) → tái tạo vector thuộc tính số từ detector; chặn đường tắt "bỏ qua tap A". Mặc định `λ_kl=λ_rec=0.1`, `T=2.0`; bật/tắt qua config.

### P1.6 — Train 2 stage
- **Stage A (alignment):** freeze LLM+MLP+encoders; train `ModalAdapter`(+`proj_penult`)+`g`. LR ~1e-4.
- **Stage B (instruction-tune nhẹ):** thêm LoRA-LLM rank 4–8, alpha 16; MLP+encoders vẫn frozen. LR ~2e-5.
- Flag `run_stage_b`: cho chạy **chỉ Stage A** (data nhỏ dễ overfit). Config: `λ_kl, λ_rec, T, lora_rank, lr theo stage, dropout, modality_dropout(0.1–0.2 augment), freeze flags`.

### P1.7 — Eval faithfulness (bắt buộc)
- **Ablation tap A:** zero/nhiễu raw token → kỳ vọng `Cues` suy giảm rõ, `Emotion` (tap B) gần như giữ nguyên. Cues không đổi ⇒ model bỏ qua tap A → tăng `λ_rec` / chọn cue trực giao hơn.
- **Agreement:** `argmax` nhãn-LLM vs `argmax` MLP — phải cao.
- **Counterfactual:** bóp một modality input → MLP đổi nhãn → kiểm lời giải thích đổi *nhất quán* không.
- **NN-decode (sanity):** chiếu soft token về token vocab gần nhất; toàn rác ⇒ activation vector, chưa có ngữ nghĩa.
- **Hedge:** trên gold, ca MLP sai thì lời giải thích có giảm chắc chắn không.

## Ràng buộc & nghiệm thu
- Phase 1: MLP + encoders không nhận gradient (assert).
- `mlp.forward` giữ tương thích ngược.
- Chỉ thêm motion cue nếu Phase 0 xong (nhất quán feature–cue).
- Smoke test end-to-end trên 1 batch; siêu tham số mới vào config có default.
- Mỗi phase: báo plan + chỗ sửa TRƯỚC khi code; sau khi code chạy smoke test + tóm tắt.

## Lưu ý data nhỏ (3399, 8 lớp)
Ưu tiên Stage A; vào Stage B chỉ LoRA rank thấp + weight decay + early stopping theo **metric eval faithfulness** (không theo train loss). Tận dụng zero-mask làm modality dropout. Generative cue cho nhiều bit/sample hơn nhãn trần ⇒ hợp data nhỏ.

---

# RÀ SOÁT & SỬA LỖI implementation LLM-1 đã viết

> Bản implement trước (`llm1_faithful_explainer_summary.md`) làm TRƯỚC khi prompt được cập nhật. Nhiều phần đúng (backward-compat, distill từ MLP, tránh output ViT-FER, GHead, 2-stage). Nhưng có các lỗi sau — sửa đúng các file đã viết, đừng làm lại từ đầu.

## FIX 1 — [NGHIÊM TRỌNG] Context cue sai hoàn toàn (`cue_extractor.py`, nhánh context)
Hiện: OpenCV → brightness / color variance / edge density trên **một peak frame**. Sai trên bốn mặt:
1. Đây là **thống kê ảnh tĩnh**, không mô tả "streamer đang làm gì". Không liên quan đập bàn / nhảy nhót.
2. Tính trên **1 frame** → mất chuyển động. Hành động chỉ tồn tại qua *chuỗi* khung.
3. **Orthogonal nhưng VÔ NGHĨA:** brightness/edge gần như không mang tín hiệu cảm xúc (facecam streamer ánh sáng ~cố định) ⇒ model sẽ học cách bỏ qua context cue → modality context vô dụng trong giải thích. (Trực giao nhãn là *cần* nhưng chưa *đủ*; cue còn phải *liên quan cảm xúc* qua đường không-rò-nhãn. Motion thoả cả hai; brightness chỉ thoả vế đầu.)
4. Phá **nhất quán feature–cue** khi context encoder đổi sang pose.

Sửa: nhánh context của `CueExtractor` phải **gate theo `context_encoder` đang active**:
- `context_encoder = pose` → `MotionCueExtractor` từ pose-kinematics: motion energy toàn thân, vận tốc/gia tốc tay, xung hướng xuống (đập bàn), năng lượng tuần hoàn (nhảy). Tính trên **chuỗi**, không peak frame. Discretize → named cue (`motion — sharp downward arm impact` / `motion — high whole-body energy, rhythmic`).
- `context_encoder = vit_imagenet` (nhánh ablation) → feature này mù hành động, nên **để trống context cue** (không bịa action cue). Không dùng brightness/edge.

**Bỏ hẳn brightness/color/edge** dù ở nhánh nào — nó sai bất kể encoder. Nguyên tắc: cue context phải khớp đúng thứ nhánh encoder active thật sự mã hoá (`pose`→chuyển động; `vit_imagenet`→không có action cue), tuyệt đối không hứa với LLM action cue mà feature không mang.

## FIX 2 — attr vector của `L_rec` (context dims)
Hiện attr 15-dim = face5 + voice3 + **context3 (brightness/color/edge)** + text4. Ba dim context đang ép context token giữ lại brightness — sai thứ. Sửa theo nhánh active: nhánh `pose` → đổi 3 dim context thành kinematic (motion energy / impact / periodicity) khớp `MotionCueExtractor`; nhánh `vit_imagenet` → **bỏ** 3 dim context khỏi attr vector (còn 12-dim), vì không có context cue để guard. Tức attr vector cũng gate theo `context_encoder`.

## FIX 3 — GHead phải tái tạo PER-MODALITY, không global pool (`training/llm1_explanation.py`)
Hiện: GHead tái tạo cả 15-dim từ **mean-pool TẤT CẢ raw token**. Vấn đề: pool toàn cục pha trộn modality, làm loãng gradient mỗi modality (~1/4) và cho phép model dồn việc cho modality dễ → không đảm bảo **từng** token modality giữ cue của *chính nó* (modality mạnh che modality yếu).
Sửa: tách theo modality — attr face tái tạo **chỉ từ face token**, voice từ audio token, motion từ context token, text từ text token (mỗi modality một head nhỏ, loss cộng theo modality). Và xác nhận GHead **chỉ** đọc 4 raw token (tap A), **không** đọc `penult` (tap B) hay `fusion` — nếu lỡ đọc penult thì nó tái tạo attr từ anchor và vô hiệu hoá cả mục đích.

## FIX 4 — Bổ sung 2 eval còn thiếu (`evaluation/faithfulness.py`)
Hiện có agreement / tap-A ablation / NN-decode (đúng spec B6 cũ). Thêm hai cái mạnh hơn trong P1.7 mới:
- **Counterfactual** (quan trọng nhất): nhiễu/loại 1 modality ở **INPUT** → MLP đổi nhãn → kiểm `Cues`+`Emotion` có đổi *nhất quán* theo hướng đó không. Nhãn đổi mà giải thích đứng im = đang biện minh, không giải thích.
- **Hedge:** trên tập gold, các ca MLP **sai** thì độ chắc chắn trong lời giải thích có giảm không.

## FIX 5 — Các điểm nhỏ
- **penult phải lấy ở `eval()`** (dropout off). `Dropout(0.3)` nằm ngay trước head, nên ở train-mode penult sẽ ngẫu nhiên → neo tap B không ổn định. MLP frozen rồi nhưng phải đảm bảo `.eval()` khi trích penult.
- **Hướng KL distill:** teacher là MLP. Dùng `KL(softmax(MLP_logits/T) ‖ LLM_dist)` (teacher‖student) theo chuẩn KD, không phải reverse. Kiểm lại trong code.
- **Face geometric cue** (EAR/MAR/brow/yaw từ MediaPipe — chấp nhận được vì dataset không có OpenFace AU) phải được **discretize thành từ** ("mouth open", "brows raised", "head turned") trước khi vào target text, không để số trần — để đẩy soft token về vùng ngôn ngữ.
- **Text là modality dễ rò nhãn nhất.** Theo dõi qua ablation: nếu bỏ text token mà `Emotion` (không chỉ `Cues`) đổi nhiều, nghĩa là LLM đang dùng từ vựng transcript làm đường tắt nhãn thay vì đọc tap B — giảm trọng số text cue hoặc trừu xuất mạnh hơn.

## Thứ tự sửa
FIX 1 + FIX 2 đi cùng Phase 0 (hoặc cùng quyết định bỏ context tạm thời). FIX 3/4/5 độc lập, làm ngay trên code hiện có. Sau sửa: chạy lại smoke test + 20 test cũ, đảm bảo không regression.

---

# FIX 6 — Rà soát Phase 0 + FIX 1–5 đã implement

> Implement (`implementation_summary_phase0_fix1to5.md`) phần lớn đúng: GHeadPerModality, gate cue/attr theo nhánh, confidence-as-signal, GRU chạy trên kinematic feature, penult ở `eval()` — giữ nguyên. Sửa các điểm sau.

## FIX 6.1 — [QUAN TRỌNG] `eval_hedge` đang đo sai khái niệm (`evaluation/faithfulness.py`)
Hiện đo hedge trên ca **MLP sai so với gold** (`mlp_pred != gold`). Sai về nguyên lý: **LLM không thấy gold**, nó chỉ đọc penult ⇒ thứ duy nhất nó có thể kế thừa là **độ bất định của MLP**, không phải tính đúng/sai. Nếu MLP **overconfident khi sai** (thường gặp khi thiếu calibration), một LLM *trung thực* buộc phải overconfident theo → `hedge_rate` thấp khi đó là hành vi **đúng**, nhưng metric lại gắn cờ "thất bại". Metric hiện trộn lẫn faithfulness-của-LLM với calibration-của-MLP.

Sửa: đổi `eval_hedge` thành đo **tương quan giữa confidence token Emotion của LLM và confidence softmax của MLP** (vd Pearson/Spearman trên toàn tập, hoặc calibration của LLM-conf theo MLP-conf). Cái này cô lập đúng câu hỏi "LLM có kế thừa độ bất định của MLP không". Có thể giữ thống kê gold-wrongness cũ như metric *phụ* về calibration của MLP, nhưng **đừng** coi nó là thước đo faithfulness của LLM.

## FIX 6.2 — PoseContextEncoder quá lớn cho data nhỏ (`encoders/context_pose.py`)
BiGRU `hidden=512, 2 lớp, bidirectional` + `Linear(1024,768)` ≈ ~8M tham số train **from scratch** (không pretrain như ViT-ImageNet frozen). Trên 3399 mẫu đây là rủi ro overfit lớn nhất của Phase 0.
Sửa: giảm dung lượng — `hidden_dim 128–256`, cân nhắc `n_layers=1`, tăng dropout (0.3–0.5) + weight decay mạnh, early stopping theo val. Đưa các số này ra config. Kiến trúc quan trọng ít hơn việc ghìm số tham số. (Việc đưa kinematic feature vào GRU đã đúng — giữ.)

## FIX 6.3 — [BUG SỐ HỌC] Keypoint mất → spike vận tốc/gia tốc giả → "đập bàn" giả (`encoders/context_pose.py` + `cue_extractor.py`)
Khi keypoint rớt rồi xuất hiện lại, velocity/acceleration giữa hai khung tạo **bước nhảy khổng lồ giả tạo** — đúng cái signal dùng để phát hiện `sharp downward arm impact`. Tracking hay rớt nhất *khi nhảy/chồm mạnh*, nên gap dễ bị đọc nhầm thành cú đập bàn. Unit test data sạch không lộ lỗi này.
Sửa: ở những khung confidence thấp, **mask hoặc hạ trọng số velocity/acceleration** (vd zero-out derivative qua các đoạn gap, hoặc nhân theo confidence của hai khung liên quan) trước khi tính cả feature lẫn motion cue. Giữ confidence như feature vẫn đúng; nhưng đạo hàm qua vùng gap phải bị vô hiệu để không giả mạo impact.

## FIX 6.4 — Xác nhận chi tiết KL (`training/llm1_explanation.py`)
`F.kl_div` có hai bẫy thầm lặng — kiểm trong code:
- Tham số đầu phải là **log-probabilities** (`log_softmax`), không phải softmax. Tên biến `lm_soft` nghi là softmax thường → xác nhận/sửa.
- Có scale loss theo **T²** không (chuẩn knowledge distillation khi dùng temperature). Nếu thiếu, thêm.

## FIX 6.5 — Bất nhất chiều kinematic (`encoders/context_pose.py`)
Input ghi `(T, K, 4)` = x,y,z,confidence, nhưng feature ghi `K×7`. Với 3D, position+velocity+acceleration+confidence = **10**; ra 7 nghĩa là đang ở 2D hoặc đã bỏ bớt. Kiểm: có lỡ **bỏ z** hay **bỏ position** không. Nếu bỏ position thì mất cue tư thế ("tay giơ trên vai") — chỉ còn động học thuần; xác nhận đây là chủ ý, không phải lỗi.

## FIX 6.6 — Điểm nhỏ
- **`eval_counterfactual`** mới kiểm `Emotion` đổi nhất quán; thêm (tuỳ chọn) kiểm **`Cues` dịch chuyển đúng** (zero audio → voice cue biến mất). Tap-A ablation gánh một phần rồi nên ưu tiên thấp.
- **Ngưỡng cứng** `impact > 0.15`, `hedge prob < 0.6`, và các bin discretize: đưa ra config và **calibrate trên phân vị tập train**, không hardcode.

---

# FIX 7 — Rà soát FIX 6 đã implement

> FIX 6 phần lớn đúng (6.1 correlation, 6.2 capacity ~570k, 6.5 dim, 6.6 config). Còn hai điểm correctness chưa xong + vài điểm kiểm.

## FIX 7.1 — [CHƯA XONG] 6.3 mới vá feature, CÒN HỞ ở cue (`cue_extractor.py`)
6.3 đã mask velocity/acceleration trong `context_pose.py::_build_kinematic_features` (đúng cho **feature** của encoder). Nhưng `MotionCueExtractor` trong `cue_extractor.py` tự tính `wrist impact` theo **đường riêng, độc lập** → cú `"sharp downward arm impact"` **vẫn bị gap tracking giả mạo**. Chính là bug ban đầu, chỉ dịch sang nhánh cue.
Sửa: áp **cùng logic confidence-masking** vào computation của `MotionCueExtractor` (đặc biệt impact/velocity), hoặc tốt hơn — refactor để cue và feature **dùng chung một hàm kinematic đã mask**, tránh hai đường tính lệch nhau. Nguyên tắc: impact chỉ được tính từ derivative ở các khung confidence đủ cao.

## FIX 7.2 — [GIẢ ĐỊNH CHỊU LỰC, CHƯA XÁC NHẬN] Mapping nhãn MLP (8 lớp) ↔ token LLM (`training/llm1_explanation.py` + `evaluation/faithfulness.py`)
Cả 6.1 (`llm_conf = prob(emotion_label_token)`) lẫn KL (`mlp_soft` vs `lm_soft`) đều giả định lấy sạch "xác suất LLM gán cho lớp k". Kiểm và sửa nếu cần:
- **Nhãn nhiều token:** nếu tên lớp nào tokenize thành >1 token (vd "disappointment", "frustration"), `prob(emotion_label_token)` mơ hồ (token đầu? tích token?) và hai nhãn trùng token đầu sẽ nhập nhằng. Đảm bảo mỗi lớp ánh xạ về **một token đại diện duy nhất** (hoặc convention nhất quán, vd dùng tích log-prob các token của tên lớp) — áp **cùng** cho cả KL lẫn 6.1.
- **KL phải cùng support:** KL so phân phối **8-d của MLP** với LLM. Phải **restrict logits LLM về đúng 8 token đại diện rồi renormalize** trước khi tính KL. Nếu đang tính trên toàn vocab vs vector 8-d → sai. Xác nhận.
- Thêm assert/test nhỏ: 8 token đại diện phân biệt nhau, và phân phối LLM-restricted sum = 1.

## FIX 7.3 — 6.4 cần test, không chỉ comment (`training/llm1_explanation.py`)
6.4 chỉ thêm comment khẳng định `log_softmax` + hướng + T². Thêm unit test bảo vệ: (a) hai phân phối giống hệt → `L_kl ≈ 0`; (b) xác nhận `T²` thật sự được nhân (so gradient/giá trị ở T=1 vs T=2). Tin nhưng phải có lưới.

## FIX 7.4 — 6.6 mới ra config, CHƯA calibrate (P0.3)
Các bin `[0.02,0.08]`, `[0.05,0.15]`… vẫn là magic number, chỉ là nằm trong config. Việc thật — tính ngưỡng theo **phân vị tập train** (vd dùng phân phối motion energy/impact thực tế) — để lại P0.3 khi có data. Ghi TODO rõ trong config để không quên.

## FIX 7.5 — Guard diễn giải cho 6.1 (`evaluation/faithfulness.py`)
Correlation chỉ có nghĩa nếu `mlp_conf` **có phương sai**. Nếu MLP overconfident bão hòa (conf ≈ 1 gần như mọi mẫu), `pearson_r` mất power và dễ đọc nhầm. Thêm: log **std/spread của `mlp_conf`** kèm correlation; nếu spread quá thấp, đánh cờ "correlation không đáng tin" thay vì báo một con r vô nghĩa.

---

# FIX 8 — Rà soát FIX 7 đã implement

> 7.3/7.4/7.5 xong (test KL, calibrate TODO, spread guard). 7.2 đúng hướng nhưng còn hở; 7.1 nhiều khả năng gate sai khung. Sửa các điểm correctness sau.

## FIX 8.1 — [CHÍ MẠNG] Binary gate impact phải WINDOW-AWARE, không kiểm một khung (`cue_extractor.py`)
Cơ chế spike giả: cổ tay track ở frame 0–4, **mất ở 5–9** (conf≈0, vị trí rác), **xuất hiện lại frame 10 với conf CAO**. `velocity[10] = pos[10] − pos[9]`, mà `pos[9]` là rác → velocity khổng lồ → impact giả. **Khung impact (frame 10) có conf cao** ⇒ nếu binary gate chỉ kiểm conf *của khung impact*, nó **PASS** và impact giả vẫn lọt.
Sửa: gate phải kiểm **mọi khung trong cửa sổ đạo hàm** — velocity cần `conf[t]` **và** `conf[t-1]` đều ≥ threshold; acceleration cần cả `conf[t], conf[t-1], conf[t-2]`. Bất kỳ khung nào trong cửa sổ dưới ngưỡng → impact tại t bị loại. (Đường feature ở 6.3 thoát được vì nhân chéo `conf[t]·conf[t-1]`; cue phải đạt hiệu quả tương đương.)

## FIX 8.2 — [CẤU TRÚC] Gộp một hàm kinematic cho cả feature lẫn cue (`context_pose.py` + `cue_extractor.py`)
Hiện feature dùng mask **nhân liên tục** (`·conf`), cue dùng **binary gate** (threshold 0.3) — hai kiểu mask khác nhau cho cùng một việc → khung conf=0.25 bị cue zero hẳn nhưng feature chỉ giảm 0.06×. Đúng cái "hai phép tính song song dễ lệch" mà 7.1 định dẹp, còn ở dạng nhẹ.
Sửa (nên làm, không bắt buộc): refactor cho cue và feature **gọi chung một hàm kinematic đã mask** (một nguồn sự thật). Nếu giữ hai đường, ít nhất **thống nhất kiểu mask** (cùng binary hoặc cùng nhân) để ngưỡng nhất quán.

## FIX 8.3 — [7.2 chưa kín] Nhãn nhiều token: cần HANDLING, không chỉ warning (`training/llm1_explanation.py`)
`_build_label_token_ids()` mới *cảnh báo* khi nhãn tokenize >1 token — nhưng cảnh báo rồi vẫn ngầm lấy token đầu thì nhập nhằng còn nguyên (hai nhãn trùng token đầu vẫn đụng). Uniqueness assertion chỉ chặn trùng token-đại-diện, không giải quyết nhãn vốn nhiều token.
Sửa: chọn một convention **quyết định** và áp nhất quán cho cả KL lẫn hedge:
- Ưu tiên: đảm bảo 8 surface string của nhãn đều **single-token** với tokenizer đang dùng (đổi tên hiển thị nếu cần, vd map "disappointment"→một token thay thế), rồi assert tất cả single-token; **hoặc**
- Dùng **tích log-prob các subtoken** của tên lớp làm score, nhất quán hai chỗ.
Không để rơi về "lấy token đầu" thầm lặng.

## FIX 8.4 — [XÁC NHẬN] KL phải restrict 8 token + renormalize (`training/llm1_explanation.py`)
Kiểm phân phối LLM đưa vào KL: phải **giới hạn logits về đúng 8 token đại diện rồi `softmax` lại trên 8 token** (sum=1 trên 8), để cùng support với phân phối 8-d của MLP. Nếu `_extract_emotion_logprobs()` trả log-prob trên **toàn vocab** mà không renormalize trên 8 token → KL đang so khác support, sai. Thêm assert: `llm_dist.sum() ≈ 1` trên đúng 8 phần tử. Dùng cùng `_build_label_token_ids()` (8.3) cho tập 8 token này.

## Sau FIX 8
Chạy lại 113 test + 3 test KL (7.3). Thêm một test cho 8.1: chuỗi keypoint có gap (conf=0 giữa chừng) **không** sinh impact cue — đây là test bắt đúng bug mà data sạch giấu đi.
