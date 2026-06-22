# Task (v2): Hỗ trợ dataset song ngữ VI+EN trong pipeline multimodal emotion (Vie-GameEmo)

> Prompt/spec cho Claude Code. Đọc HẾT trước khi sửa. Giữ pattern config-driven,
> tuân thủ phần "Ràng buộc & KHÔNG được làm", và mỗi task chỉ xong khi đạt acceptance.
> v2 bổ sung so với v1: encoder CafeBERT, language-adversarial head (thay cho fusion
> conditioning cũ), focal loss + mixup, iterative multilabel stratification,
> fastText cross-check, ma trận ablation mở rộng + fragmentation diagnostic.

---

## 0. Bối cảnh

**Vie-GameEmo**: multimodal emotion recognition cho video game livestream, 4 modality:
- Audio (spectrogram → AST) → `h_audio` (64, 768)
- Visual face (ViT-FER) → `h_face` (1, 768)
- Visual context (ViT gameplay) → `h_ctx` (1, 768)
- Text (ASR transcript → text encoder) → `h_text` (1, 768)

Fuse bằng Conv-Attention Pre-Fusion → MLP → 8 nhãn: `neutral, hype, amused, tilted,
sad, shocked, fear, disgusted`. Downstream: 4 LLM Reasoner xuất reasoning **tiếng Việt**.

**Dataset**: ~3,245 clip × 5s, **class imbalance nặng** (neutral 43% … disgusted ~0.5%,
đang bổ sung lên ~100–150). **Tỷ lệ ngôn ngữ: EN ≈ 50–60% (đồng-đa số); tiếng Việt là
phe THIỂU SỐ (~40–50%).** EN được thêm để bù class hiếm + genre ít video Việt (horror,
RPG), nhưng vì chiếm 50–60% nên EN trải rộng cả class phổ biến.

**Chỉ text modality bị ảnh hưởng bởi ngôn ngữ.** Audio/face/context language-agnostic — KHÔNG sửa logic.

### Hai rủi ro then chốt (định hướng thiết kế)
- **R1 — Tiếng Việt thiểu số bị under-served:** encoder thấy nhiều EN hơn → nhánh VI có thể yếu đi. Đây là rủi ro chính ở chế độ EN-dominant.
- **R2 — Confound ngôn ngữ↔nhãn (nhẹ nhưng cần kiểm):** EN có thể overrepresent *bên trong* các class hiếm → model học lối tắt "EN ⟹ class hiếm" qua text branch. Mức độ chỉ biết được sau report phân bố per-class × language.

---

## 1. Nguyên tắc thiết kế

1. **Tách 2 nhánh transcript:**
   - **A — Classifier** (`transcript → encoder → h_text → fusion → MLP`): GIỮ NGUYÊN ngôn
     ngữ gốc, encode bằng encoder multilingual fine-tune end-to-end, **KHÔNG dịch**.
   - **B — LLM Reasoner / Consolidator**: để LLM tự hiểu transcript EN, xuất reasoning VI
     qua prompt song ngữ pin tiếng Việt. **KHÔNG model dịch riêng.**
2. **Biết ngôn ngữ ở cấp VIDEO (metadata-first), không đoán ở cấp clip.** Whisper-detect +
   fastText chỉ để **validate/audit**, không phải đường đi chính (clip 5s detect không ổn).
3. **Encoder ưu tiên CafeBERT** (XLM-R-large + pretrain tiếng Việt) để bảo vệ phe VI thiểu
   số (R1), kèm fallback nhẹ hơn vì ràng buộc VRAM T4.
4. **Language-invariance bằng adversarial head** (gradient-reversal): ép `h_text` KHÔNG mã
   hóa ngôn ngữ → giảm phân mảnh embedding (R1) và chặn lối tắt confound (R2). Đây là
   đóng góp/novelty chính. Bật/tắt qua config; quyết định bật dựa trên ablation.
5. **Config-driven tuyệt đối** — mọi switch qua `config.yaml`, không hardcode.
6. **Output cuối luôn tiếng Việt.** **Dịch chỉ là nhánh ABLATION**, không phải production.

---

## 2. Các task

### TASK 2.1 — `data/schemas.py`: trường ngôn ngữ + audit

```python
from typing import Literal

class Annotation(BaseModel):
    clip_id: str
    emotion_label: EmotionLabel
    transcript: str
    code_switching_ratio: float
    source_language: Literal["vi", "en"]          # ground-truth cấp video (metadata)
    asr_detected_language: str | None = None        # Whisper info.language
    text_detected_language: str | None = None        # fastText lid.176 trên transcript
    language_detect_confidence: float | None = None   # prob của fastText
    language_mismatch: bool = False                   # bất kỳ detector ≠ source_language
```
**Acceptance:** validate được; backfill `source_language="vi"` cho clip cũ; clip VI cũ load không lỗi.

---

### TASK 2.2 — `config.yaml`: ASR routing metadata-first + cross-check

```yaml
annotation:
  asr:
    backend: "whisper"               # whisper | phowhisper
    language_routing: "metadata"      # metadata (mặc định) | auto | force
    force_language: "vi"
    detect_for_validation: true        # luôn chạy detect để audit, KHÔNG ép kết quả
    lang_prob_threshold: 0.6           # chỉ dùng khi routing == auto
    text_lid:
      backend: "fasttext"             # lid.176 (offline, <1MB) cross-check transcript
      model: "lid.176.ftz"
    whisper:
      model: "large-v3"
      vad_filter: true                 # giảm hallucinate trên im lặng
      vi: { language: "vi", initial_prompt: "<gaming prompt VI: GG, clutch, ace, ơi trời, vãi...>", post_process: "bartpho" }
      en: { language: "en", initial_prompt: "<gaming prompt EN: GG, clutch, ace, let's go, oh my god, no way...>", post_process: "none" }
    phowhisper:
      model: "vinai/PhoWhisper-large"
      vi: { language: "vi", post_process: "bartpho" }
      # không có "en" → clip EN fallback sang whisper
```
**Acceptance:** đọc được; clip VI không đổi hành vi.

---

### TASK 2.3 — `data/annotator/whisper_asr.py`: routing + EN + cross-check

Sửa `WhisperASR`, `PhoWhisperASR`, `build_asr()`:
1. `transcribe(audio_path, source_language=None)`:
   - `routing=="metadata"`: dùng `whisper[source_language]` (language + prompt + post_process).
   - `routing=="auto"`: bỏ `language` cho Whisper tự detect, giới hạn cân nhắc {vi,en}; dưới
     `lang_prob_threshold` → default an toàn `vi` (PhoWhisper/whisper-vi).
   - `routing=="force"`: ép `force_language`.
   - Luôn ghi `asr_detected_language` (= `info.language`) và set `language_mismatch` nếu khác `source_language`. KHÔNG ép theo detect.
2. **fastText lid.176** chạy trên transcript đầu ra → `text_detected_language` + `language_detect_confidence`; nếu khác source_language → cờ mismatch (để review code-switch).
3. `PhoWhisperASR` gọi với `source_language=="en"` → log warning + **fallback `WhisperASR`** (không crash).
4. `BARTphoPostProcessor` chỉ chạy khi `post_process=="bartpho"` (clip VI). Clip EN bỏ qua.
5. Bật `vad_filter=True`.

**Acceptance:** clip VI y hệt hiện tại; clip EN cho transcript tiếng Anh sạch (không ký tự VI rác), prompt gaming EN, không BARTpho; mọi clip điền đủ field detect + mismatch; phowhisper+EN tự fallback.

---

### TASK 2.4 — LLM prompts: bilingual-aware (KHÔNG dịch trước)

Sửa template trong `consolidator.py`, `llm/llm1_explainer.py`, `llm/llm2_coreasoner.py`,
(và `llm3_vlm.py`, `llm4_rlvr.py` nếu chèn transcript). Mỗi prompt:
- Thêm chỉ dẫn: transcript có thể VI **hoặc** EN → hiểu trực tiếp, KHÔNG dịch trong câu trả lời, reasoning **100% tiếng Việt** (pin rõ "Trả lời hoàn toàn bằng tiếng Việt").
- Truyền thêm `{source_language}` vào template.
- KHÔNG thêm bước translate transcript; KHÔNG route pipeline riêng cho EN.

**Acceptance:** transcript EN vẫn cho reasoning VI mạch lạc; transcript VI output không đổi.

---

### TASK 2.5 — Split: iterative multilabel stratification + report (chống R1/R2)

1. Dùng **`iterative-stratification` (trent-b), `MultilabelStratifiedKFold`** với joint label
   `(emotion, source_language, genre, codeswitch_bucket)`. (Tránh quirk của `scikit-multilearn iterative_train_test_split`.)
2. **In report phân bố** sau split: bảng `emotion × source_language` cho từng split → đo trực tiếp mức confound R2 (tỷ lệ EN bên trong từng class).
3. Class hiếm (`disgusted/fear/shocked`) phải có mặt ở cả 3 split; với `disgusted` (~16–150): dùng **k-fold CV riêng** và báo cáo mean±std (test đơn ~16 clip quá nhiễu). Đảm bảo mỗi cell `(rare-class × language)` có số tối thiểu, hoặc ghi rõ chỗ không thể.

```yaml
split:
  method: "iterative_multilabel"   # trent-b MultilabelStratifiedKFold
  stratify_on: [emotion, source_language, genre, codeswitch_bucket]
  rare_class_cv: true
```
**Acceptance:** report in ra; không class hiếm nào vắng split; phân bố VI/EN cân trong từng class (hoặc lý do được ghi).

---

### TASK 2.6 — Encoder factory: CafeBERT default + options (bảo vệ phe VI thiểu số)

`encoders/text_xlmr.py` + `build_text_encoder()`:
```yaml
encoders:
  text:
    backend: "cafebert"   # cafebert(default) | xlmr-large | xlmr-base | xlm-emo | mE5-frozen | phobert
    model: "uitnlp/CafeBERT"
    warm_start: "none"     # "MilaNLProc/xlm-emo-t" để test transfer emotion-pretrain
    freeze: false          # true cho mE5-frozen baseline (rẻ compute)
```
- **Default CafeBERT** (XLM-R-large + pretrain VI; SOTA VLUE gồm VSMEC, vẫn đa ngôn ngữ).
- **Fallback `xlmr-base`** nếu T4 hết VRAM (xem caveat compute) — interface `encode()` không đổi.
- Options ablation: `xlm-emo` (warm-start emotion-pretrain), `mE5-frozen` (multilingual-e5-large, frozen + L2/z-score norm — rẻ), `phobert` (VI-only, chỉ cho nhánh ablation translated/VI-only — PhoBERT không encode EN).

**Acceptance:** switch encoder qua config không sửa code khác; CafeBERT là default; có đường fallback base.

---

### TASK 2.7 — Language-adversarial head (NOVELTY, thay cho fusion-conditioning v1)

> Thay thế hoàn toàn ý "cho fusion biết source_language" của v1 (ý đó reinforce confound).
> Hướng đúng: ép `h_text` **không** dự đoán được ngôn ngữ.

- Thêm một **language discriminator** nhỏ (MLP 2 lớp) nhận `h_text`, dự đoán `source_language` (vi/en), nối qua **Gradient Reversal Layer (GRL)** với hệ số `lambda_grl`.
- Loss tổng: `L = L_emotion + lambda_grl * L_lang_adv` (GRL đảo dấu gradient của nhánh ngôn ngữ khi backprop vào encoder).
- Đặt sau cờ config; mặc định **tắt** để giữ baseline so sánh được:
```yaml
encoders:
  text:
    language_adversarial:
      enabled: false        # bật theo quyết định ablation (xem dưới)
      lambda_grl: 0.1
```
- **Quyết định bật:** chỉ bật nếu ablation A2 (mixed, no-adv) cho thấy nhánh **VI test subset** thấp hơn A1 (VI-only) >~1–2 macro-F1, hoặc fragmentation diagnostic (2.9-E) cho thấy phân tách ngôn ngữ rõ trong `h_text`.

**Acceptance:** tắt cờ → fusion/training y hệt baseline; bật cờ → train ổn định (theo dõi loss không phân kỳ), language-classification-accuracy từ `h_text` giảm về gần ngẫu nhiên.

---

### TASK 2.8 — Class imbalance (ghép với động lực bù class hiếm)

`config.yaml`:
```yaml
loss:
  type: "focal"            # focal(ưu tiên) | weighted_ce | ce
  gamma: 2.0
  class_weights: "inverse_freq"   # alpha theo prior
sampler: "balanced_batch"  # tùy chọn
augment:
  fused_mixup: true        # Manifold/Multimodal Mixup ở mức fused-embedding cho class hiếm
```
- Mặc định **focal loss (γ=2) + inverse-freq weights**; theo dõi để focal+weights không bóp class đa số (neutral) — so với weighted-CE, giữ cái nào cho **rare-class recall** cao hơn mà không tụt macro-F1.
- **Mixup ở mức fused embedding** (không phải SMEM raw) để sinh cặp (feature,label) ảo cho class hiếm; tránh trộn cảm xúc mâu thuẫn.
- **Báo cáo macro-F1 + per-class recall + confusion matrix**, KHÔNG headline accuracy (neutral 43% làm accuracy gây hiểu lầm).

**Acceptance:** loss/sampler/mixup switch qua config; metric rare-class được log riêng.

---

### TASK 2.9 — Ablation scaffolding (lõi học thuật) — chạy qua config, không sửa core

**A. Language-handling (headline):**
A1 VI-only (CafeBERT/PhoBERT) · A2 Mixed multilingual no-translate (**đề xuất**) · A3 Mixed + adversarial head (**novelty**) · A4 Translate EN→VI (NLLB-200)→PhoBERT · A5 Translate all→EN→English emotion model.
**B.** F1 tách riêng **test-VI** và **test-EN** (chứng minh phe VI thiểu số không bị bóp — R1).
**C.** Per-class F1 cho disgusted/fear/shocked qua A1–A5 (EN bù có thật sự cứu class hiếm?).
**D.** Modality ablation: T / A / V_face / V_ctx riêng; cặp; cả 4. Gồm câu hỏi "text có giúp hơn A+V không?" và "khác nhau theo ngôn ngữ không?".
**E. Fragmentation diagnostic:** t-SNE/UMAP `h_text` tô màu theo ngôn ngữ, **trước vs sau** fine-tune (và trước/sau adversarial), kèm metric định lượng (accuracy của language-classifier từ frozen `h_text`, hoặc silhouette theo ngôn ngữ).

Translation chỉ tồn tại ở A4/A5 (sau cờ `text.mode: translate_vi|translate_en`; model `facebook/nllb-200-distilled-1.3B`).

**Acceptance:** A1–A5 chạy được qua config; B/C/D/E tính & log; diagnostic E xuất hình + số.

---

## 3. Ràng buộc & KHÔNG được làm
- KHÔNG dịch ở nhánh classifier production (chỉ A4/A5).
- KHÔNG model dịch riêng cho LLM Reasoner — prompt song ngữ pin VI.
- KHÔNG auto-detect làm primary cho clip 5s; **metadata-first**, detect chỉ audit.
- KHÔNG để fusion "nhìn thấy" `source_language` (mâu thuẫn mục tiêu adversarial, reinforce confound).
- KHÔNG sửa logic audio/face/context encoder; KHÔNG sửa fusion core (chỉ thêm adversarial head ở nhánh text).
- KHÔNG đổi default behaviour clip VI (regression phải pass).
- Adversarial head + mixup mặc định **tắt**, bật theo quyết định ablation.
- **VRAM T4 16GB:** CafeBERT là XLM-R-large (~560M) + 3 modality + fusion → đo VRAM thực tế; nếu OOM dùng gradient checkpointing / batch nhỏ / freeze tầng dưới, hoặc fallback `xlmr-base`. ĐO trước khi cam kết.
- **ĐO trên clip của chính bạn**, đừng tin các con số WER/BLEU/accuracy/LID từ tài liệu ngoài (dataset-dependent).

---

## 4. Thứ tự thực hiện
1. 2.1 → 2.2 → 2.3 (schema/config/ASR) — lõi, test kỹ + regression VI.
2. 2.4 (LLM prompts) — song song được.
3. 2.5 (split + report) — quan trọng cho tính hợp lệ; report tiết lộ mức R2.
4. 2.6 (encoder CafeBERT + đo VRAM) → 2.8 (imbalance).
5. 2.9 (ablation A1/A2 trước) → nếu cần thì 2.7 (adversarial A3).

## 5. Định nghĩa "Done"
- [ ] Clip VI: regression pass, hành vi không đổi.
- [ ] Clip EN: transcript sạch; LLM xuất reasoning VI mạch lạc.
- [ ] `source_language` + field detect/mismatch điền đủ; vào stratification.
- [ ] Split iterative multilabel; report `emotion×language` per split in ra; class hiếm có ở mọi split (hoặc CV riêng).
- [ ] Encoder mặc định CafeBERT, có fallback base; switch qua config.
- [ ] Focal loss + per-class recall/confusion matrix được log (không headline accuracy).
- [ ] Adversarial head bật/tắt qua config, tắt = baseline; ablation A1–A5 + diagnostic E chạy được.
- [ ] Không core modality nào (audio/face/context/fusion-core) bị đổi logic.
