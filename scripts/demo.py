"""Gradio demo for Vie-GameEmo.

Two modes:
    --mode batch:    upload a video file, get prediction + reasoning
    --mode realtime: webcam or screen capture, sliding-window inference

Usage:
    # Perception only (default; runs with just perception_best.pt):
    python scripts/demo.py --config config.yaml --checkpoint web_demo/perception_best.pt --mode batch

    # With LLM reasoning (needs an LLM adapter checkpoint configured in config.yaml):
    python scripts/demo.py --config config.yaml --checkpoint web_demo/perception_best.pt --mode batch --enable-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vie_gameemo.utils.config import load_config
from vie_gameemo.utils.logging import setup_logging
from vie_gameemo.utils.seed import set_seed

logger = logging.getLogger(__name__)

_EMOTION_LABELS = ["neutral", "hype", "amused", "tilted", "sad", "shocked", "fear", "disgusted"]
_EMOTION_COLORS = {
    "neutral":   "#95A5A6",
    "hype":      "#FF6B35",
    "amused":    "#27AE60",
    "tilted":    "#C0392B",
    "sad":       "#5D6D7E",
    "shocked":   "#8E44AD",
    "fear":      "#34495E",
    "disgusted": "#7D6608",
}
_EMOTION_EMOJI = {
    "neutral":   "😐",
    "hype":      "🔥",
    "amused":    "😄",
    "tilted":    "😤",
    "sad":       "😢",
    "shocked":   "😱",
    "fear":      "😨",
    "disgusted": "🤢",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["batch", "realtime"], default="batch")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Share via Gradio public URL")
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM reasoning (requires an LLM adapter checkpoint). "
             "Off by default so the demo runs with only the perception checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=Path(cfg.logging.file))
    set_seed(cfg.seed)

    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Gradio not installed. Install with: pip install gradio"
        ) from exc

    if args.mode == "batch":
        demo = _build_batch_demo(cfg, args)
    else:
        demo = _build_realtime_demo(cfg, args)

    logger.info("Launching Gradio demo on port %d (share=%s)", args.port, args.share)
    # Enable the queue so generator functions stream intermediate updates
    # (per-batch timeline rows) to the browser as they are produced.
    demo.queue()
    # Gradio 6 moved `css` from Blocks(...) to launch(...).
    demo.launch(server_port=args.port, share=args.share, css=_DEMO_CSS)
    return 0


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def _build_batch_demo(cfg, args):
    """Gradio Blocks app: upload video → prediction + reasoning."""
    import gradio as gr

    def predict_video(video_path: str | None) -> tuple[str, str, str]:
        """Process uploaded video and return prediction results."""
        if video_path is None:
            return "No video uploaded", "", ""

        tmp_out = Path(tempfile.mktemp(suffix=".json"))
        try:
            from vie_gameemo.inference.batch import batch_inference
            batch_inference(
                clip_paths=[Path(video_path)],
                checkpoint=args.checkpoint,
                cfg=cfg,
                output_json=tmp_out,
                include_llm_explanation=args.enable_llm,
                use_cached_features=False,
            )
            results = json.loads(tmp_out.read_text(encoding="utf-8"))
            if not results:
                return "Error: no results", "", ""

            r = results[0]
            if "error" in r:
                return f"Error: {r['error']}", "", ""

            label = r.get("predicted_label", "neutral")
            confidence = r.get("confidence", 0.0)
            reasoning = r.get("reasoning", "")
            scores = r.get("class_scores", {})

            emoji = _EMOTION_EMOJI.get(label, "❓")
            label_html = (
                f"<div style='text-align:center; font-size:2em;'>"
                f"{emoji} <b>{label.upper()}</b> ({confidence:.1%})"
                f"</div>"
            )

            score_lines = [
                f"{_EMOTION_EMOJI.get(k, '')} **{k}**: {v:.2%}"
                for k, v in sorted(scores.items(), key=lambda x: -x[1])
            ]
            scores_md = "\n".join(score_lines)

            if not args.enable_llm:
                reasoning = "_LLM reasoning disabled (run with --enable-llm)._"
            return label_html, scores_md, reasoning or "_Reasoning not available_"
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return f"Error: {exc}", "", ""
        finally:
            if tmp_out.exists():
                tmp_out.unlink(missing_ok=True)

    llm_note = "" if args.enable_llm else " _(LLM reasoning off — perception only)_"
    with gr.Blocks(title="Vie-GameEmo — Game Streamer Emotion Recognition") as demo:
        gr.Markdown(
            "# Vie-GameEmo Demo\n"
            "Upload a Vietnamese game livestream clip to analyze the streamer's emotion."
            + llm_note
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Clip (mp4)")
                predict_btn = gr.Button("Analyze Emotion", variant="primary")
            with gr.Column(scale=1):
                label_output = gr.HTML(label="Predicted Emotion")
                scores_output = gr.Markdown(label="Class Scores")
                reasoning_output = gr.Textbox(
                    label="LLM Reasoning", lines=6, interactive=False,
                    visible=args.enable_llm,
                )

        predict_btn.click(
            fn=predict_video,
            inputs=[video_input],
            outputs=[label_output, scores_output, reasoning_output],
        )
        gr.Examples(
            examples=[],
            inputs=[video_input],
            label="Example clips (add paths here)",
        )

    return demo


# ---------------------------------------------------------------------------
# Real-time mode
# ---------------------------------------------------------------------------

# Disable Gradio's orange "generating" highlight and style the status banner.
# In Gradio 6 `css` is passed to launch(), not the Blocks constructor.
_DEMO_CSS = """
.generating, .pending { border-color: transparent !important;
    box-shadow: none !important; animation: none !important; }
.progress-bar, .meta-text-center { display: none !important; }
.vg-status { font-size: 0.95em; opacity: 0.75; margin: 2px 0; }
.vg-err { color: #dc2626; opacity: 1; }
"""


def _fmt_ts(sec: float) -> str:
    """Format seconds as M:SS."""
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def _status_html(text: str, state: str = "busy") -> str:
    """Render a minimal, low-key status line."""
    return f'<div class="vg-status vg-{state}">{text}</div>'


def _timeline_rows(history: list[dict]) -> str:
    """Format prediction history as a Markdown table.

    A Markdown component (unlike gr.Dataframe) reliably re-renders on every
    generator yield, so the table grows live as each batch finishes.
    """
    header = (
        "| Đoạn (thời gian) | Cảm xúc chủ đạo | Độ tin cậy |\n"
        "|---|---|---|\n"
    )
    if not history:
        return header + "| _(chưa có)_ | | |"
    lines = []
    for i, item in enumerate(history, 1):
        label = item.get("label", "?")
        emoji = _EMOTION_EMOJI.get(label, "❓")
        start = item.get("start_sec", 0)
        end = item.get("end_sec", 0)
        conf = item.get("confidence", 0)
        lines.append(
            f"| #{i}  {_fmt_ts(start)} – {_fmt_ts(end)} | {emoji} {label} | {conf:.0%} |"
        )
    return header + "\n".join(lines)


def _timeline_summary(history: list[dict]) -> str:
    """Build a dominant-emotion + distribution summary from history."""
    if not history:
        return ""
    from collections import Counter
    counts = Counter(h.get("label", "?") for h in history)
    total = len(history)
    dom, dom_n = counts.most_common(1)[0]
    lines = [
        f"### {_EMOTION_EMOJI.get(dom, '')} Cảm xúc chủ đạo: **{dom}** "
        f"({dom_n}/{total} đoạn)",
        "",
        "**Phân bố cảm xúc theo các đoạn:**",
    ]
    for label, n in counts.most_common():
        bar = "█" * max(1, round(20 * n / total))
        lines.append(f"- {_EMOTION_EMOJI.get(label, '')} `{label:<9}` {bar} {n / total:.0%}")
    return "\n".join(lines)


def _build_realtime_demo(cfg, args):
    """Gradio Blocks app: drop a long video → per-window emotion timeline."""
    import gradio as gr

    from vie_gameemo.inference.realtime import RealtimeInferenceRunner

    # Fixed settings: 5s batches, audio always on.
    _BATCH_SECONDS = 5.0

    def analyze_timeline(video_path: str | None):
        """Generator: yields (status, timeline_rows, summary) per 5s batch.

        Runs in Gradio's worker thread and yields after each batch so the UI
        renders that row before the next (heavy) batch is computed.
        """
        if video_path is None:
            yield _status_html("Chưa có video — thả một clip vào ô bên trái.", "err"), _timeline_rows([]), ""
            return

        yield _status_html("Đang nạp mô hình nhận diện cảm xúc… (lần đầu ~30–60s)"), _timeline_rows([]), ""

        # Non-overlapping batches: window == step so each segment is analysed once.
        runner = RealtimeInferenceRunner(
            checkpoint=args.checkpoint,
            cfg=cfg,
            window_seconds=_BATCH_SECONDS,
            step_seconds=_BATCH_SECONDS,
            skip_llm=True,
            drop_slow_windows=False,  # keep every batch when analysing a file
            use_audio=True,
        )

        yield _status_html("Đang phát hiện vùng webcam & tách âm thanh…"), _timeline_rows([]), ""

        history: list[dict] = []
        try:
            for pred in runner.iter_predictions(Path(video_path)):
                history.append(pred)
                t = _fmt_ts(pred.get("start_sec", 0))
                t2 = _fmt_ts(pred.get("end_sec", 0))
                yield (
                    _status_html(
                        f"Đang phân tích… xong đoạn {len(history)} ({t}–{t2}). "
                        f"Đang xử lý đoạn tiếp theo…"
                    ),
                    _timeline_rows(history),
                    _timeline_summary(history),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Timeline analysis failed: %s", exc)
            yield (
                _status_html(f"Lỗi: {exc}", "err"),
                _timeline_rows(history),
                _timeline_summary(history),
            )
            return

        yield (
            _status_html(f"Xong — đã phân tích {len(history)} đoạn (mỗi đoạn 5s).", "done"),
            _timeline_rows(history),
            _timeline_summary(history),
        )

    with gr.Blocks(title="Vie-GameEmo — Dòng thời gian cảm xúc") as demo:
        gr.Markdown("# Vie-GameEmo — Phân tích cảm xúc theo đoạn")
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Thả video vào đây")
                analyze_btn = gr.Button("Phân tích (mỗi đoạn 5s)", variant="primary")
                status = gr.HTML(_status_html("Sẵn sàng.", "done"))
            with gr.Column(scale=2):
                summary_md = gr.Markdown()
                gr.Markdown("### Kết quả theo từng đoạn")
                timeline_df = gr.Markdown(_timeline_rows([]))

        analyze_btn.click(
            fn=analyze_timeline,
            inputs=[video_input],
            outputs=[status, timeline_df, summary_md],
        )

    return demo


if __name__ == "__main__":
    sys.exit(main())
