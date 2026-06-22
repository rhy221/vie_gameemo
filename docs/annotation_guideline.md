# Vie-GameEmo Annotation Guideline (`gaming_8`)

**Schema:** `gaming_8` — 8 nhãn cảm xúc phù hợp livestream game / Vietnamese game-streaming livestreams.
**Version:** 1.1 (2026-05)
**Áp dụng cho / Applies to:** clip ~5 giây, có / không có webcam streamer.

> **Thay đổi từ v1.0:** Bỏ nhãn `focus` — các clip tryhard/concentration không biểu cảm rõ ràng được gán `neutral`. Cập nhật clip duration ~5 giây.

---

## 0. Quy ước chung / General Conventions

**VI.**
- Mỗi clip nhận **đúng một** nhãn — cảm xúc *chủ đạo* trong cửa sổ ±1.5 giây quanh **peak frame** (frame có AU intensity cao nhất, xem `peak_frame.py`).
- Đối với clip có nhiều cảm xúc chuyển tiếp nhanh (ví dụ shocked → amused), chọn cảm xúc **chiếm thời lượng dài nhất** trong clip; nếu hòa nhau, ưu tiên cảm xúc tại peak frame.
- Khi không chắc chắn, dùng **decision tree** ở §3. Không bao giờ gán `neutral` chỉ để "an toàn" — `neutral` có định nghĩa riêng (§1.1).
- Annotator độc lập gán nhãn, sau đó tính Cohen's κ. Yêu cầu κ ≥ 0.6 trước khi đưa clip vào dataset.

**EN.**
- Each clip receives **exactly one** label — the *dominant* emotion in the ±1.5 s window around the **peak frame** (frame with highest AU intensity, see `peak_frame.py`).
- For clips with rapid emotion transitions (e.g., shocked → amused), pick the emotion with the **longest duration** in the clip; ties go to the peak frame.
- When unsure, use the **decision tree** in §3. Never pick `neutral` as a "safe default" — it has a specific definition (§1.1).
- Two annotators label independently, then Cohen's κ is computed. Require κ ≥ 0.6 before including a clip in the dataset.

**Multimodal cues considered / Tín hiệu đa phương thức xem xét:**
1. Face (Action Units, FACS): AU4/6/12/20/25/26… intensities at the peak frame.
2. Prosody: pitch (Hz), RMS energy (dB), shouting, laughing, silence.
3. Transcript: lexical content (Vietnamese + English gaming slang).
4. Game context: in-game event (clutch, defeat, jump-scare, gore, joke).

---

## 1. Định nghĩa nhãn / Label Definitions

> Order = class index (0..7). Phải khớp với `EmotionLabel` enum trong `src/vie_gameemo/data/schemas.py`.

### 1.1 `neutral` (idx 0) — *Trung tính / Baseline*

**VI.**
Trạng thái nền: streamer giải thích cơ chế game, đọc chat, chờ matchmaking, thao tác bình thường, hoặc **tập trung im lặng** (tryhard không kèm biểu cảm rõ ràng).
- **Khuôn mặt:** AU intensity thấp toàn bộ (max AU < 1.0); mắt mở bình thường (AU45 không kích hoạt mạnh). AU4 có thể nhích nhẹ khi tập trung nhưng không nổi bật.
- **Giọng:** pitch trung bình, năng lượng đều, tốc độ nói bình thường, không có tiếng cười/hét/im lặng đột ngột.
- **Transcript:** câu mô tả/giải thích ("ờ thì cái này…", "build này là…"), command ngắn không kèm cảm xúc ("đi", "lùi", "ok ok"), không có exclamation.
- **Game context:** không có sự kiện kích thích (match đang chờ, đang đi tới objective, đang xem chat, đang tryhard không phản ứng).

**EN.**
Baseline state: the streamer is explaining game mechanics, reading chat, waiting for matchmaking, performing normal actions, or **silently concentrating** (tryhard with no visible affect).
- **Face:** low AU intensity overall (max AU < 1.0); eyes normally open (AU45 not strongly activated). AU4 may be slightly active during concentration but not prominent.
- **Voice:** medium pitch, even energy, normal pace; no laughter, shouting, or sudden silence.
- **Transcript:** explanatory or descriptive sentences ("uhm, the thing is…", "this build is…"), short commands without affect ("go", "ok ok"), no exclamations.
- **Game context:** no stimulating event (lobby, traveling to objective, reading chat, tryhard with no emotional reaction).

✅ **Positive example:** Streamer giải thích build cho người mới, giọng đều, AU đều thấp.
✅ **Positive example:** Streamer im lặng, nhìn màn hình, chơi ranked — AU4 nhích nhẹ nhưng không có biểu cảm rõ.
❌ **Negative example:** Streamer "cười nhẹ" khi nói chuyện → đó là `amused`, không phải `neutral`.
❌ **Negative example:** Streamer im lặng nhưng mặt **rõ ràng bực bội** (AU4 rất cao, AU23/24) → `tilted`.

---

### 1.2 `hype` (idx 1) — *Phấn khích / Adrenaline rush*

**VI.**
Bùng nổ năng lượng tích cực: clutch thành công, ace, pentakill, victory royale, jump-scare survived, level-up clutch.
- **Khuôn mặt:** AU12 (zygomatic, cười) **cao**, AU6 (cheek raiser) cao, AU26 (jaw drop) — *smile mở miệng*; mắt mở to (AU5).
- **Giọng:** **shouting** (RMS > -10 dB), pitch tăng vọt, "POG!", "LET'S GO!", "ăn rồi!", "WOOOO".
- **Transcript:** exclamation, gaming slang dương tính ("clutch", "ez", "GG", "POG", "ăn pentakill", "ace luôn").
- **Game context:** sự kiện win/clutch/highlight vừa xảy ra (kill cuối, headshot, level up đúng lúc).

**EN.**
Positive energy burst: clutch win, ace, pentakill, victory royale, surviving a jump-scare, clutch level-up.
- **Face:** AU12 (zygomatic, smile) **high**, AU6 (cheek raiser) high, AU26 (jaw drop) — *open-mouth smile*; eyes wide (AU5).
- **Voice:** **shouting** (RMS > -10 dB), pitch spike, "POG!", "LET'S GO!", "ăn rồi!", "WOOOO".
- **Transcript:** exclamation, positive gaming slang ("clutch", "ez", "GG", "POG", "got the pentakill", "ace").
- **Game context:** win/clutch/highlight just occurred (final kill, headshot, well-timed level-up).

⚠️ **vs `amused`:** hype = high arousal + positive (post-clutch); amused = lower arousal + positive (laughing at a joke or fail).
⚠️ **vs `shocked`:** hype is sustained celebration; shocked is the brief surprise *before* it resolves to hype.

---

### 1.3 `amused` (idx 2) — *Vui thích / Laughter*

**VI.**
Vui vẻ, cười (không bùng nổ): nghe joke, gặp bug funny, teammate troll, fail hài.
- **Khuôn mặt:** AU12 cao (smile), AU6 cao (cheek raiser); AU25/26 (open mouth) khi cười to; mắt nheo (AU7 nhẹ do AU6).
- **Giọng:** **laughing** (laughter detected by audio agent), pitch dao động, RMS trung-cao nhưng *không phải* shout.
- **Transcript:** "haha", "buồn cười quá", "trời ơi", "thằng này ngu thật".
- **Game context:** moment hài (fail funny, troll, bug, NPC ngớ ngẩn, teammate chơi ngu).

**EN.**
Joy / laughter (not explosive): hearing a joke, finding a funny bug, teammate trolling, hilarious fail.
- **Face:** AU12 high (smile), AU6 high (cheek raiser); AU25/26 (open mouth) during loud laughter; squinted eyes (AU7 from AU6).
- **Voice:** **laughter** detected by audio agent, pitch varies, RMS medium-high but *not* shouting.
- **Transcript:** "haha", "lol", "so funny", "what a clown".
- **Game context:** funny moment (funny fail, troll, bug, silly NPC, throwing teammate).

⚠️ **vs `hype`:** amused = laughter; hype = celebratory shouting. Khi cười + hét cùng lúc, ưu tiên cảm xúc đang chiếm 60%+ thời lượng.

---

### 1.4 `tilted` (idx 3) — *Tức giận / Anger, frustration*

**VI.**
Bực bội, tức giận do thua/lag/team kém: ragequit, chửi teammate, đập bàn, complain.
- **Khuôn mặt:** AU4 (cau mày) **rất cao**, AU7 (lid tighten), AU23/24 (lip tight/press), đôi khi AU17 (chin raiser); KHÔNG có AU12.
- **Giọng:** **shouting** với pitch cao + RMS cao (nhưng KHÁC hype — tone tiêu cực), thở mạnh, có thể có giọng nghẹn.
- **Transcript:** chửi thề, slang tiêu cực ("đm", "trash team", "lag vl", "feed", "int", "throw", "noob"), exclamation tiêu cực.
- **Game context:** bị thua/feed/lag, teammate troll, bị gank liên tục, mất Baron/objective.

**EN.**
Anger, frustration from losing/lag/bad team: ragequit, flaming teammates, table-slap, complaining.
- **Face:** AU4 (brow lowerer) **very high**, AU7 (lid tighten), AU23/24 (lip tight/press), occasionally AU17 (chin raiser); NO AU12.
- **Voice:** **shouting** with high pitch + high RMS (DIFFERENT from hype — negative tone), heavy breathing, possibly cracking voice.
- **Transcript:** profanity, negative slang ("dmm", "trash team", "lag af", "feed", "int", "throw", "noob"), negative exclamations.
- **Game context:** losing/feeding/lag, teammate trolling, repeated ganks, lost Baron/objective.

⚠️ **vs `hype`:** Cả hai đều shout. Khác biệt là **valence**: hype = positive (win), tilted = negative (loss/lag). Đọc transcript + game context để quyết định.
⚠️ **vs `sad`:** tilted = explosive anger; sad = quiet resignation. Nếu im lặng và "ờ thua rồi" → `sad`.
⚠️ **vs `neutral`:** tilted im lặng (silent rage) vẫn phân biệt được qua AU4 **rất cao** + AU23/24 + game context loss. Nếu AU4 nhẹ + không có loss event → `neutral`.

---

### 1.5 `sad` (idx 4) — *Buồn / Sadness, regret*

**VI.**
Buồn, thất vọng nhẹ, hối tiếc: thua match căng nhưng không hề nổi nóng, mất item rare, nhân vật game chết (storyline), kết thúc series buồn.
- **Khuôn mặt:** AU1 + AU4 (inner brow raise + brow lowerer — *sad combo*), AU15 (lip corner depressor — *frown*), AU17, đôi khi AU43 (eyes closed).
- **Giọng:** giọng trầm, RMS thấp, tốc độ nói chậm, thở dài, *không* shout.
- **Transcript:** "thôi rồi", "tiếc thật", "buồn ghê", "haizz", "lại thua nữa", không có chửi thề.
- **Game context:** sự kiện cảm xúc trong cốt truyện (NPC chết, ending buồn), thua match căng, mất rank.

**EN.**
Sadness, mild disappointment, regret: losing a tight match without rage, losing a rare item, narrative death, sad series ending.
- **Face:** AU1 + AU4 (inner brow raise + brow lowerer — *sad combo*), AU15 (lip corner depressor — *frown*), AU17, occasionally AU43 (eye closure).
- **Voice:** low pitch, low RMS, slow speech, sighs, *no* shouting.
- **Transcript:** "oh well", "sad", "haizz", "lost again", no profanity.
- **Game context:** narrative emotional event (NPC death, sad ending), losing a close match, losing rank.

⚠️ **vs `tilted`:** sad is quiet; tilted is loud. Nếu chửi thề mạnh hoặc đập bàn → `tilted`, không phải `sad`.

---

### 1.6 `shocked` (idx 5) — *Sốc / Surprise*

**VI.**
Bất ngờ ngắn (1-3 giây): teammate clutch bất ngờ, kill outplay, jump-scare nhẹ, plot twist, "wait what?".
- **Khuôn mặt:** AU1 + AU2 (brow raise + outer brow raise — *eyebrows shoot up*), AU5 (upper lid raise — *wide eyes*), AU26/27 (jaw drop / mouth stretch).
- **Giọng:** giật mình ngắn ("wHAT?", "ơ?", "huh?"), pitch tăng đột ngột, có thể ngắt giữa câu.
- **Transcript:** "ơ kìa", "wait what", "ơ ơ ơ", "thật á?", "wtf" (trung tính, không tức), exclamation question.
- **Game context:** sự kiện đột ngột mới xảy ra trong 1-2 giây trước (sniper headshot, plot twist, secret).

**EN.**
Brief surprise (1-3 s): unexpected teammate clutch, kill outplay, mild jump-scare, plot twist, "wait what?".
- **Face:** AU1 + AU2 (brow raise + outer brow raise — *eyebrows shoot up*), AU5 (upper lid raise — *wide eyes*), AU26/27 (jaw drop / mouth stretch).
- **Voice:** brief startle ("wHAT?", "huh?", "no way?"), sudden pitch rise, possibly mid-sentence cut.
- **Transcript:** "wait what", "no way", "really?", neutral "wtf" (surprise, not angry), exclamation questions.
- **Game context:** sudden event in the last 1-2 s (sniper headshot, plot twist, secret reveal).

⚠️ **vs `hype`:** shocked là *trước khi* não xử lý — kéo dài ngắn. Nếu sau 2 giây streamer tiếp tục hét "LET'S GO" → cảm xúc *chủ đạo* là `hype`. Chỉ chọn `shocked` khi clip kết thúc trong trạng thái surprise.
⚠️ **vs `fear`:** shocked = surprise valence trung tính/tích cực; fear = surprise có valence âm + tránh né.

---

### 1.7 `fear` (idx 6) — *Sợ hãi / Fear, panic*

**VI.**
Sợ hãi rõ rệt (chủ yếu game horror, survival, jump-scare mạnh): Resident Evil, Phasmophobia, Outlast, ARMA night ops.
- **Khuôn mặt:** AU1 + AU2 + AU4 (raised + drawn brows — *fear brow*), AU5 (wide eyes), AU7, AU20 (lip stretcher — *grimace*), AU26 (jaw drop kéo dài).
- **Giọng:** tiếng la sợ hãi (khác hype: pitch rất cao, run rẩy), thì thầm "shhhh", "đừng…", hoặc im lặng nín thở.
- **Transcript:** "trời ơi", "đừng nhe đừng", "chạy chạy chạy", "no no no", "what is that", thì thầm.
- **Game context:** game horror với threat hiện diện, jump-scare vừa xảy ra, NPC địch xuất hiện đột ngột, không gian tối.

**EN.**
Pronounced fear (mostly horror, survival, hard jump-scares): Resident Evil, Phasmophobia, Outlast, ARMA night ops.
- **Face:** AU1 + AU2 + AU4 (raised + drawn brows — *fear brow*), AU5 (wide eyes), AU7, AU20 (lip stretcher — *grimace*), AU26 (sustained jaw drop).
- **Voice:** frightened scream (DIFFERENT from hype: very high pitch, trembling), whispered "shhhh", "don't…", or silent breath-holding.
- **Transcript:** "oh god", "no no no", "run run run", "what is that", whispering.
- **Game context:** horror game with active threat, just-occurred jump-scare, sudden enemy appearance, dark environment.

⚠️ **vs `shocked`:** fear là sợ kéo dài + tránh né; shocked là bất ngờ thoáng qua + valence trung tính.
⚠️ **vs `tilted`:** Cả hai shout, nhưng fear có pitch cao + run; tilted có giọng giận + chửi thề.
⚠️ **Genre hint:** Nếu game không phải horror/survival, hiếm khi gán `fear`. Trong MOBA/FPS, "panic" thường là `tilted` hoặc `shocked`.

---

### 1.8 `disgusted` (idx 7) — *Ghê tởm / Disgust, contempt, cringe*

**VI.**
Ghê tởm hoặc khinh thường: gore quá đáng, nhân vật xấu, content cringe, teammate chơi tệ đến mức "ngao ngán", ý kiến viewer toxic.
- **Khuôn mặt:** AU9 (nose wrinkler — *nhăn mũi*), AU10 (upper lip raiser), AU17 (chin raise), AU25, đôi khi AU16 (lower lip depressor).
- **Giọng:** "ờ trời", "ghê quá", "thôi xin", "yikes", tone hạ thấp / chán nản, không shout (trừ khi gore quá mạnh).
- **Transcript:** "ghê quá", "đùa á", "cringe", "ew", "thôi xin lỗi", "nhìn không nổi", "cái gì đây".
- **Game context:** gore, body horror, character xấu xí, animation kì cục, teammate ngu lặp lại, joke phản cảm.

**EN.**
Disgust, contempt, or cringe: excessive gore, ugly characters, cringe content, teammate so bad it's "facepalm", toxic viewer comments.
- **Face:** AU9 (nose wrinkler), AU10 (upper lip raiser), AU17 (chin raise), AU25, occasionally AU16 (lower lip depressor).
- **Voice:** "ugh", "gross", "no thanks", "yikes", lowered/exasperated tone, no shouting (unless extreme gore).
- **Transcript:** "ew", "cringe", "no", "what even", "I can't watch", "what is this".
- **Game context:** gore, body horror, ugly characters, weird animations, repeatedly stupid teammate, distasteful joke.

⚠️ **vs `tilted`:** disgusted = revulsion ("ew, gross"); tilted = anger ("WTF this is bullshit"). AU9/AU10 (nose wrinkle) là phân biệt then chốt — disgust mới có.
⚠️ **vs `sad`:** disgusted kéo môi trên lên (AU10); sad kéo môi xuống (AU15).

---

## 2. Phân bổ mục tiêu / Target Distribution

**VI.** Mục tiêu 600 clip, cân bằng theo tier hiếm-thường:

| # | Nhãn       | %    | Clips | Tier        | Ghi chú thu thập                                     |
|---|------------|------|-------|-------------|------------------------------------------------------|
| 0 | neutral    | 20%  | 120   | common      | Lobby, đọc chat, giải thích build, tryhard im lặng  |
| 1 | hype       | 15%  | 90    | common      | Win clip, ace, pentakill (chủ yếu MOBA/FPS)          |
| 2 | amused     | 13%  | 78    | common      | Funny moments, troll, bug clip                        |
| 3 | tilted     | 13%  | 78    | medium      | Ragequit, flame, lag complaint                        |
| 4 | sad        | 11%  | 66    | medium      | Mất rank, ending buồn, mất item rare                 |
| 5 | shocked    | 12%  | 72    | medium      | Plot twist, unexpected clutch, sniper kill            |
| 6 | fear       | 9%   | 54    | rare        | **Tăng share game horror lên ≥ 18%**                 |
| 7 | disgusted  | 7%   | 42    | rare        | Gore, cringe, body horror, toxic comments             |
|   | **Total**  | **100%** | **600** | | |

**EN.** Target 600 clips, balanced by rarity tier:

| # | Label      | %    | Clips | Tier   | Sourcing note                                        |
|---|------------|------|-------|--------|------------------------------------------------------|
| 0 | neutral    | 20%  | 120   | common | Lobby, reading chat, build explanations, silent tryhard |
| 1 | hype       | 15%  | 90    | common | Win clips, ace, pentakill (mainly MOBA/FPS)           |
| 2 | amused     | 13%  | 78    | common | Funny moments, trolls, bug clips                      |
| 3 | tilted     | 13%  | 78    | medium | Ragequit, flame, lag complaints                       |
| 4 | sad        | 11%  | 66    | medium | Rank loss, sad ending, lost rare item                 |
| 5 | shocked    | 12%  | 72    | medium | Plot twist, unexpected clutch, sniper kill            |
| 6 | fear       | 9%   | 54    | rare   | **Raise horror genre share to ≥ 18%**                 |
| 7 | disgusted  | 7%   | 42    | rare   | Gore, cringe, body horror, toxic comments             |
|   | **Total**  | **100%** | **600** | | |

**Genre rebalance for fear/disgusted:**
- Horror genre share trong `config.yaml: data.genre_distribution.horror` nên ở mức **≥ 15%** (hiện tại) hoặc tăng lên **18-20%** nếu thấy fear/disgusted không đủ.
- RPG genre có thể đóng góp `sad` (storyline) và `disgusted` (gore).
- Mobile / casual hiếm có fear/disgusted → tập trung crawl cho 6 nhãn còn lại.
- **neutral tăng lên 20%** để bao gồm cả các clip tryhard/concentration trước đây được gán `focus`.

---

## 3. Decision Tree khi mơ hồ / Disambiguation Decision Tree

```
Bắt đầu / Start:
│
├─ Có shout (RMS > -10 dB)? / Shouting?
│   ├─ Yes:
│   │   ├─ Lexical valence positive (POG, GG, win, clutch)?
│   │   │   ├─ Yes → HYPE
│   │   │   └─ No:
│   │   │       ├─ Profanity / blame / loss event?
│   │   │       │   ├─ Yes → TILTED
│   │   │       │   └─ No:
│   │   │       │       ├─ Pitch very high + trembling + horror genre?
│   │   │       │       │   ├─ Yes → FEAR
│   │   │       │       │   └─ No → SHOCKED (recheck: ngắn < 3s?)
│   └─ No:
│       ├─ Laughing detected?
│       │   ├─ Yes → AMUSED
│       │   └─ No:
│       │       ├─ AU9/AU10 active (nose wrinkle / lip raise)?
│       │       │   ├─ Yes → DISGUSTED
│       │       │   └─ No:
│       │       │       ├─ AU1+AU4 + AU15 (sad combo) + low pitch?
│       │       │       │   ├─ Yes → SAD
│       │       │       │   └─ No:
│       │       │       │       ├─ AU4 rất cao + AU23/24 + loss event (không shout)?
│       │       │       │       │   ├─ Yes → TILTED (silent rage)
│       │       │       │       │   └─ No → NEUTRAL
```

**Lưu ý quan trọng / Important:**
- Decision tree là **gợi ý**, không bắt buộc. Luôn đối chiếu với **3+ modality** trước khi quyết định.
- Nếu sau khi đi qua tree vẫn không chắc → mark `_uncertain` và đưa lên human review queue.
- **Tryhard/concentration clips** (trước đây là `focus`) → mặc định gán `neutral` trừ khi có biểu cảm rõ.

---

## 4. Edge Cases & Quy tắc đặc biệt / Special Rules

**VI.**

1. **Webcam bị che / không có:** Nếu `webcam_bbox = None`, chỉ dùng audio + transcript + game context để quyết định. Đánh dấu `human_verified=true` bắt buộc cho các clip này.
2. **Streamer đeo mask / kính dày:** AU extraction kém chất lượng → ưu tiên audio + transcript. Ghi chú vào `metadata.notes`.
3. **Two emotions back-to-back trong clip:**
   - Quy tắc 60/40: nếu một cảm xúc chiếm ≥ 60% thời lượng → chọn nó.
   - Nếu hòa (50/50): chọn cảm xúc *gần peak frame* hơn.
   - Nếu vẫn không phân định được → chia clip thành 2 clip riêng, mỗi clip một nhãn.
4. **Sarcasm / mỉa mai:** "ez game ez life" sau khi *thua* → `tilted`, không phải `hype`. Đọc game context kỹ.
5. **Acting / scripted content:** Nếu streamer đang đóng vai (ASMR horror, comedy skit), gán cảm xúc *được biểu diễn*, không phải cảm xúc thật. Note vào `metadata.is_acted=true`.
6. **Code-switching VI-EN:** Slang tiếng Anh (clutch, POG, throw, int) giữ nguyên trong transcript. Annotator phải hiểu nghĩa — xem glossary §5.
7. **Tiếng chửi thề Việt:** "dmm", "đm", "vcl", "vl" — coi như marker tilted nếu kết hợp với loss event. Trong context vui (joke giữa bạn bè) có thể là `amused`.
8. **NPC face nhầm với streamer:** Khi `webcam_bbox` không tin cậy (stability_score < 0.5), giả định AU đến từ NPC → bỏ qua AU, dùng audio + transcript.

**EN.**

1. **Webcam occluded / missing:** If `webcam_bbox = None`, decide based on audio + transcript + game context only. Force `human_verified=true` for these clips.
2. **Streamer wearing mask / thick glasses:** Poor AU extraction → prioritize audio + transcript. Note in `metadata.notes`.
3. **Two emotions back-to-back in one clip:**
   - 60/40 rule: if one emotion occupies ≥ 60% of duration → pick it.
   - 50/50 tie: pick the emotion *closer to the peak frame*.
   - Still indistinguishable → split the clip into two clips, one label each.
4. **Sarcasm:** "ez game ez life" *after* a loss → `tilted`, not `hype`. Read game context carefully.
5. **Acting / scripted content:** If the streamer is roleplaying (ASMR horror, comedy skit), label the *performed* emotion, not the actor's true state. Note `metadata.is_acted=true`.
6. **Code-switching VI-EN:** Keep English gaming slang in transcript verbatim. Annotators must understand the slang — see glossary §5.
7. **Vietnamese profanity:** "dmm", "đm", "vcl", "vl" — treat as tilted marker if combined with a loss event. In a friendly-banter context it may be `amused`.
8. **NPC face confused with streamer:** When `webcam_bbox` is unreliable (`stability_score < 0.5`), assume AUs come from an NPC → ignore AUs, use audio + transcript.

---

## 5. Glossary — Gaming slang VI-EN

| Term            | Meaning (VI)                            | Likely emotion        |
|-----------------|------------------------------------------|----------------------|
| clutch          | xoay chuyển tình thế (1vN thắng)        | hype                  |
| ace             | hạ cả đội địch                          | hype                  |
| pentakill       | giết 5 (MOBA)                           | hype                  |
| POG / pogchamp  | phấn khích, sốc tích cực                | hype / shocked        |
| GG / GGWP       | good game (kết thúc)                    | neutral / hype / sad  |
| ez              | dễ                                       | hype hoặc sarcasm     |
| int / inting    | cố tình feed (chơi tệ)                  | tilted                |
| feed            | bị giết quá nhiều                        | tilted / sad          |
| throw           | làm hỏng game                            | tilted                |
| tilted          | bực, mất tập trung                       | tilted                |
| sweaty / tryhard| chơi quá nghiêm túc                      | neutral               |
| jump-scare      | hù bất ngờ (horror)                      | fear / shocked        |
| cringe          | ngại / sởn da                            | disgusted             |
| salty           | cay cú sau thua                          | tilted / sad          |
| lit / based     | tốt, ấn tượng                            | hype / amused         |
| griefer         | troll teammate                           | (gây tilted cho người khác) |
| stomp           | thắng áp đảo                             | hype                  |
| smurf           | tài khoản ẩn rank                        | tilted (gặp smurf)    |

---

## 6. Quy trình annotation chính thức / Official Annotation Workflow

**VI.**
1. **Pilot (50 clips):** 2 annotator gán độc lập → tính Cohen's κ.
2. Nếu **κ < 0.6** → review disagreements, refine guideline (cập nhật version trong file này), lặp lại pilot.
3. Nếu **κ ≥ 0.6** → tiếp tục annotate phần còn lại theo tỉ lệ §2.
4. Mỗi 100 clip → re-check 10 clip ngẫu nhiên để tránh drift.
5. Sau khi hoàn thành, chạy `notebooks/01_kaggle_stage0_prepare_data.ipynb` (CELL 16) để xem `actual % vs target %` — flag bất kỳ nhãn nào lệch > 5% so với target.

**EN.**
1. **Pilot (50 clips):** 2 annotators label independently → compute Cohen's κ.
2. If **κ < 0.6** → review disagreements, refine the guideline (bump version in this file), re-run pilot.
3. If **κ ≥ 0.6** → proceed with the rest using the §2 distribution.
4. Every 100 clips → spot-check 10 random clips to prevent drift.
5. After completion, run `notebooks/01_kaggle_stage0_prepare_data.ipynb` (CELL 16) to see `actual % vs target %` — flag any label off by > 5% from target.

---

## 7. Liên kết / References

- Schema enum: `src/vie_gameemo/data/schemas.py` → `class EmotionLabel`.
- Config: `config.yaml` → `labeling.schemas.gaming_8`, `labeling.class_distribution`.
- Peak frame logic: `src/vie_gameemo/data/annotator/peak_frame.py`.
- AU reference: Facial Action Coding System (FACS), Ekman & Friesen (1978).
- OpenFace AU keys: AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26, AU45.

---

*Version 1.1 — 2026-05. Removed `focus` label (merged into `neutral`; tryhard/concentration clips → neutral). Clip duration updated to ~5 giây. Bump version on any rule change so cached annotations can be re-validated.*
