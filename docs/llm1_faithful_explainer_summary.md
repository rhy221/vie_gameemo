# LLM-1 Faithful Explainer — Tóm tắt Implementation

> Implement lại LLM-1 (Post-hoc Explainer) theo spec `claude_code_prompt.md`, thêm 2 "tap" để LLM giải thích trung thực với quyết định của MLP và giàu chi tiết mô tả, không ảnh hưởng LLM-2/3/4.

## Vấn đề gốc

LLM-1 cũ là **zero-shot**: nhận nhãn MLP rồi viết giải thích bênh vực — kể cả khi MLP sai (post-hoc rationalization), không phải reasoning thật.

## Giải pháp: 2 tap

| Tap | Nguồn | Vai trò |
|-----|-------|---------|
| **Tap B** (faithfulness anchor) | Vector 256-d áp chót của MLP (trước `Linear(256,8)`) | Buộc LLM đọc đúng quyết định phân loại |
| **Tap A** (richness source) | Raw modality tokens (audio/face/context/text) | Chi tiết mịn mà MLP đã nén bỏ |

LLM được train sinh: `"Cues: {cues}. Emotion: {label}."` — phần Cues buộc đọc tap A, phần Emotion buộc đọc tap B. **Label lấy từ MLP argmax, không dùng gold.**

## Các file đã thay đổi

### Sửa (5 files)

| File | Thay đổi |
|------|---------|
| [`src/vie_gameemo/classifiers/mlp.py`](../src/vie_gameemo/classifiers/mlp.py) | `forward(..., return_penultimate=False)` — expose vector 256-d trước head cuối. Backward-compatible (state_dict keys giữ nguyên). |
| [`src/vie_gameemo/llm/modal_adapter.py`](../src/vie_gameemo/llm/modal_adapter.py) | Thêm `proj_penult: Linear(256, d_llm)` + param `d_penult`. `forward(..., penult=None)` — nếu có, prepend vào soft token: `[penult \| fusion \| audio \| face \| ctx \| text]`. `penult=None` → hành vi cũ y nguyên. |
| [`src/vie_gameemo/llm/llm1_explainer.py`](../src/vie_gameemo/llm/llm1_explainer.py) | Hỗ trợ **2 mode**: zero-shot (cũ) và trained (mới). Tự detect mode dựa vào checkpoint có `proj_penult` hay không. `parse_output()` parse được cả format `Cues:.../Emotion:...` và `<think>/<answer>` cũ. Vẫn **không bao giờ override label**. |
| [`config.yaml`](../config.yaml) | Thêm `training.llm1_explanation` (hyperparams 2-stage training) và `llm.llm1.explanation_checkpoint` (path tới checkpoint đã train). |
| [`src/vie_gameemo/inference/batch.py`](../src/vie_gameemo/inference/batch.py) | `_load_llm()` tự detect `explanation_checkpoint` trong config. `_forward()` gọi `classifier(..., return_penultimate=True)` và đưa `penult` vào evidence dict khi inference. |

### Mới (5 files)

| File | Nội dung |
|------|---------|
| [`src/vie_gameemo/llm/cue_extractor.py`](../src/vie_gameemo/llm/cue_extractor.py) | `CueExtractor` — sinh cue **label-orthogonal** cho cả 4 modality:<br>• **Face**: MediaPipe FaceMesh → Eye/Mouth Aspect Ratio, brow height, head yaw/pitch (vì dataset không có OpenFace AU, chỉ có MediaPipe crop)<br>• **Voice**: librosa → pitch (F0), energy (RMS), speaking rate<br>• **Context**: OpenCV → brightness, color variance, edge density (trên peak frame)<br>• **Text**: regex → exclamation, negative lexicon, game terms, word count<br>Có `precompute_all()` để cache trước khi train (`data/cache/{face_geo,prosody,visual_stats}/`). Attr vector tổng 15 chiều (5+3+3+4) cho loss `L_rec`. |
| [`src/vie_gameemo/training/llm1_explanation.py`](../src/vie_gameemo/training/llm1_explanation.py) | Training 2-stage:<br>**Stage A** — freeze LLM+MLP+encoders+fusion, train `ModalAdapter` (gồm `proj_penult`) + `GHead`.<br>**Stage B** (optional) — thêm LoRA rank 4-8 vào LLM, tiếp tục train.<br>Loss: `L = L_LM + λ_kl·KL(LLM‖MLP) + λ_rec·smooth_l1(g(z_raw), attr_vec)`.<br>`GHead`: MLP nhỏ tái tạo attr_vector từ mean-pool raw token — chặn shortcut bỏ qua tap A.<br>Có modality dropout augmentation (zero ngẫu nhiên 1 modality/sample). |
| [`scripts/train_llm1.py`](../scripts/train_llm1.py) | CLI: `--stage a\|b\|both`, `--precompute-cues`, `--resume-from` (perception checkpoint). |
| [`src/vie_gameemo/evaluation/faithfulness.py`](../src/vie_gameemo/evaluation/faithfulness.py) | 3 eval bắt buộc theo spec B6:<br>1. **Agreement** — so `argmax(LLM Emotion)` vs `argmax(MLP)`<br>2. **Tap A ablation** — zero raw token, kỳ vọng Cues đổi nhưng Emotion giữ nguyên<br>3. **NN-decode** — soft token → top-3 nearest-neighbor vocab embedding (sanity check) |
| [`scripts/eval_faithfulness.py`](../scripts/eval_faithfulness.py) | CLI chạy 3 eval trên, xuất `outputs/results/faithfulness_eval.json`. |

### Test mới

[`tests/test_llm1_explanation.py`](../tests/test_llm1_explanation.py) — 20 test cases:
- `EmotionClassifier.forward(return_penultimate=True)` đúng shape, đúng giá trị (penult → head cuối khớp logits)
- `ModalAdapter` với/không `penult` — đúng shape, backward-compat
- `CueExtractor` cho cả 4 modality, fallback khi thiếu data
- `GHead` shape
- `parse_output()` cho cả 2 format

## Đảm bảo không ảnh hưởng LLM-2/3/4

| Thay đổi | Tác động lên LLM-2/3/4 |
|----------|------------------------|
| `EmotionClassifier.forward(return_penultimate=False)` mặc định | Không đổi — caller cũ không truyền flag |
| `ModalAdapter.__init__(d_penult=256)` thêm `proj_penult` | Checkpoint cũ load bằng `strict=False` → weight mới random init, không lỗi |
| `ModalAdapter.forward(penult=None)` mặc định | Không đổi — LLM-2/3/4 không bao giờ truyền `penult` |
| Training/eval script mới | File riêng biệt, không gọi từ pipeline cũ |
| Config section mới | Nằm dưới `training.llm1_explanation`, các stage khác không đọc tới |

## Verification đã chạy

- ✅ 20/20 test mới pass
- ✅ 9/9 test classifier cũ pass (không regression)
- ✅ Checkpoint `ModalAdapter` cũ (không có `proj_penult`) load thành công qua `from_checkpoint(strict=False)`
- ✅ Lời gọi kiểu LLM-2/3/4 (`adapter(fusion_emb, audio=...)`, không có `penult`) hoạt động đúng, shape không đổi
- ✅ Syntax check toàn bộ file mới + file phụ thuộc (`llm2_coreasoner.py`, `llm3_vlm.py`, `llm4_rlvr.py`, `cognition.py`, `batch.py`)
- ✅ `config.yaml` parse hợp lệ với pyyaml

## Chưa làm (cần GPU + dataset thật)

- Chạy thật `python scripts/train_llm1.py --stage a` trên dữ liệu để train `ModalAdapter` + `GHead`
- Chạy `python scripts/eval_faithfulness.py` để lấy số liệu agreement/ablation thực tế, đánh giá xem tap B có thực sự "anchor" được nhãn và tap A có thực sự cần để sinh Cues hay không
