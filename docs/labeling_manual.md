# Hướng Dẫn Dán Nhãn Thủ Công — Vie-GameEmo

**Dành cho:** Annotator dán nhãn clip bằng Label Studio  
**Schema:** `gaming_8` — 8 nhãn (xem chi tiết: `annotation_guideline.md`)  
**Clip:** ~5 giây, có hoặc không có webcam streamer

---

## Bước 1 — Cài đặt Label Studio

```bash
pip install label-studio
label-studio start
# Mở http://localhost:8080
```

**Import project:**
1. Tạo project mới → chọn template **Video Classification**
2. Import file JSON annotation hiện có (nếu có)
3. Labels cần có đúng 8 nhãn (theo thứ tự):

```
neutral | hype | amused | tilted | sad | shocked | fear | disgusted
```

---

## Bước 2 — Quy trình cho mỗi clip

Với mỗi clip ~5 giây, thực hiện theo thứ tự:

```
1. Xem toàn bộ clip (5 giây)
2. Nhìn khuôn mặt streamer → AU nào nổi bật?
3. Nghe giọng → shout / cười / thì thầm / bình thường?
4. Đọc transcript (nếu có) → từ khóa gì?
5. Nhìn màn hình game → sự kiện gì xảy ra?
6. Chọn nhãn → Submit
```

---

## Bước 3 — Bảng nhãn nhanh

| Nhãn | Idx | Nhận dạng nhanh | Keyword transcript |
|------|-----|-----------------|-------------------|
| **neutral** | 0 | Mặt bình thản, giọng đều, không sự kiện | "ờ thì", "build này", "ok ok" |
| **hype** | 1 | Miệng mở rộng + cười + HÉT | "POG!", "LET'S GO!", "ăn rồi!", "WOOOO" |
| **amused** | 2 | Cười (không hét), mắt nheo | "haha", "buồn cười", "trời ơi" |
| **tilted** | 3 | Nhăn mặt bực + HÉT tiêu cực | "đm", "trash team", "lag vl", "noob" |
| **sad** | 4 | Môi xuống + giọng trầm + không hét | "haizz", "tiếc thật", "thôi rồi" |
| **shocked** | 5 | Mắt mở to + miệng há + giật mình ngắn | "ơ kìa?", "wait what", "thật á?" |
| **fear** | 6 | Run rẩy + la hét cao giọng + tránh né | "đừng!", "chạy!", "no no no" |
| **disgusted** | 7 | Nhăn mũi + kéo môi trên lên | "ghê quá", "cringe", "ew", "yikes" |

---

## Bước 4 — Câu hỏi 3 bước khi phân vân

Khi không chắc, trả lời 3 câu hỏi:

### Q1: Streamer có hét không? (RMS cao, nghe rõ ràng)
- **Có hét + vui vẻ** (GG, POG, thắng) → **`hype`**
- **Có hét + tức giận** (chửi thề, mất game) → **`tilted`**
- **Có hét + run rẩy + game horror** → **`fear`**
- **Giật mình ngắn** (< 3 giây, "ơ?") → **`shocked`**

### Q2: Nếu không hét — có cười không?
- **Nghe tiếng cười** → **`amused`**

### Q3: Nếu không cười — nhìn khuôn mặt + game context:
- **Nhăn mũi** (AU9) → **`disgusted`**
- **Môi xuống + giọng trầm** → **`sad`**
- **Mặt bình thường / tryhard im lặng** → **`neutral`**
- **Mặt bực (AU4 rất cao) + thua/lag nhưng không hét** → **`tilted`** (silent rage)

---

## Bước 5 — Trường hợp hay nhầm lẫn

### `neutral` vs `tilted` (im lặng)
| | neutral | tilted (silent) |
|--|---------|-----------------|
| Mày | thả lỏng hoặc nhích nhẹ | **cau sâu** (AU4 rất cao) |
| Môi | bình thường | mím chặt (AU23/24) |
| Game | không sự kiện | vừa thua / lag / bị gank |
| Transcript | mô tả bình thường | không nói hoặc nói nhỏ bực |

### `hype` vs `tilted` (cả hai hét)
| | hype | tilted |
|--|------|--------|
| Cười | **Có** (AU12) | Không |
| Transcript | GG, POG, thắng | chửi thề, thua |
| Sau clip | tiếp tục vui | tiếp tục bực |

### `hype` vs `shocked`
| | hype | shocked |
|--|------|---------|
| Thời lượng | kéo dài ≥ 3s | thoáng qua ≤ 3s |
| Có cười | Có | Không nhất thiết |
| Clip kết thúc | đang ăn mừng | đang ngạc nhiên |

### `shocked` vs `fear`
| | shocked | fear |
|--|---------|------|
| Valence | trung tính / tích cực | tiêu cực / sợ |
| Sau sự kiện | "ơ kìa?", tò mò | tránh né, "đừng!" |
| Game genre | bất kỳ | chủ yếu horror |

### `sad` vs `disgusted`
| | sad | disgusted |
|--|-----|-----------|
| Môi | kéo xuống (AU15) | kéo lên (AU10) |
| Mũi | bình thường | nhăn (AU9) |
| Cause | thua, mất rank | gore, cringe, teammate tệ |

---

## Bước 6 — Các nhãn đặc biệt

### Khi nào gán `neutral` cho clip tryhard?
Clip streamer im lặng, tập trung cao, HP thấp, ranked → **`neutral`** nếu:
- Không có biểu cảm rõ trên mặt
- Không hét, không cười, không thở dài
- Chỉ nói command ngắn ("đi", "ok ok") hoặc im hoàn toàn

→ Đây là trường hợp **neutral/concentration**. Chỉ chuyển sang nhãn khác nếu có biểu cảm rõ (vd: bắt đầu hét → `hype`/`tilted`).

### Khi nào gán `_uncertain`?
- Không xác định được sau 3 lần xem lại
- Webcam bị che toàn bộ và transcript rỗng
- Cảm xúc chuyển liên tục, không nhãn nào chiếm 60%

Trong Label Studio: thêm tag `uncertain` và chuyển sang clip tiếp theo.

---

## Bước 7 — Workflow Label Studio

### Phím tắt khuyến nghị

| Phím | Nhãn |
|------|------|
| `1` | neutral |
| `2` | hype |
| `3` | amused |
| `4` | tilted |
| `5` | sad |
| `6` | shocked |
| `7` | fear |
| `8` | disgusted |
| `u` | đánh dấu uncertain (nếu cấu hình) |

### Tốc độ mục tiêu
- **Clip dễ** (neutral/hype/tilted rõ ràng): ~20-30 giây/clip
- **Clip trung bình**: ~45 giây/clip
- **Clip khó** (uncertain): 60-90 giây rồi skip

---

## Bước 8 — Kiểm tra chất lượng

### Trước khi submit batch:
1. Đếm số lượng mỗi nhãn → so với target §2 trong `annotation_guideline.md`
2. Xem lại 5-10% clip ngẫu nhiên
3. Đảm bảo không có nhãn nào chiếm > 30% (dấu hiệu bias)

### Cohen's κ với annotator 2:
- κ ≥ 0.8: xuất sắc
- κ 0.6–0.8: tốt, có thể tiếp tục
- κ < 0.6: cần review lại disagreements trước khi tiếp tục

### Disagreement thường gặp → hướng giải quyết:

| Nhãn A | Nhãn B | Tiebreaker |
|--------|--------|------------|
| neutral | tilted | AU4 intensity + có loss event không? |
| hype | shocked | Kéo dài bao lâu sau giật mình? |
| amused | hype | Có tiếng hét không? |
| sad | tilted | Có chửi thề / mất bình tĩnh không? |
| shocked | fear | Game genre + có tránh né không? |

---

## Phụ lục — Ví dụ cụ thể theo game

| Tình huống | Nhãn |
|------------|------|
| MOBA: ace pentakill + "YESSSS LET'S GO!" | **hype** |
| MOBA: bị gank, im lặng, mặt bực | **tilted** |
| MOBA: thua ranked, thở dài "haizz thôi rồi" | **sad** |
| FPS: headshot bất ngờ "ơ ơ ơ wait what?" < 3s | **shocked** |
| FPS: clutch 1v4 "ĂN RỒI ƠIIIIII!" | **hype** |
| Horror: jump-scare "ĐỪNG ĐỪNGgg TRỜI ƠI" run giọng | **fear** |
| Horror: thấy gore "ghê quá, nhìn không nổi" nhăn mũi | **disgusted** |
| Casual: bug funny, cười thoải mái "haha vãi cái game" | **amused** |
| Lobby: giải thích build, giọng bình thường | **neutral** |
| Ranked: im lặng, nhìn màn hình, không biểu cảm | **neutral** |
| Teamfight: im lặng nhưng mặt rõ ràng bực (AU4 cao) + vừa thua | **tilted** |

---

*Xem chi tiết định nghĩa từng nhãn, edge cases và quy trình inter-annotator agreement tại `docs/annotation_guideline.md`.*
