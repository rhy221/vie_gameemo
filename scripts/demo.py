"""Gradio demo for Vie-GameEmo.

Two modes:
    --mode batch:    upload a video file, get prediction + reasoning
    --mode realtime: webcam or screen capture, sliding-window inference

Usage:
    python scripts/demo.py --config config.yaml --checkpoint outputs/checkpoints/best.pt --mode batch
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

_EMOTION_LABELS = ["hype", "tilted", "focused", "disappointed", "shocked", "amused", "neutral"]
_EMOTION_COLORS = {
    "hype": "#FF6B35",
    "tilted": "#C0392B",
    "focused": "#2980B9",
    "disappointed": "#7F8C8D",
    "shocked": "#8E44AD",
    "amused": "#27AE60",
    "neutral": "#95A5A6",
}
_EMOTION_EMOJI = {
    "hype": "🔥",
    "tilted": "😤",
    "focused": "🎯",
    "disappointed": "😞",
    "shocked": "😱",
    "amused": "😄",
    "neutral": "😐",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["batch", "realtime"], default="batch")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Share via Gradio public URL")
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
    demo.launch(server_port=args.port, share=args.share)
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
                include_llm_explanation=True,
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

            return label_html, scores_md, reasoning or "_Reasoning not available_"
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return f"Error: {exc}", "", ""
        finally:
            if tmp_out.exists():
                tmp_out.unlink(missing_ok=True)

    with gr.Blocks(title="Vie-GameEmo — Game Streamer Emotion Recognition") as demo:
        gr.Markdown(
            "# Vie-GameEmo Demo\n"
            "Upload a Vietnamese game livestream clip to analyze the streamer's emotion."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Clip (mp4)")
                predict_btn = gr.Button("Analyze Emotion", variant="primary")
            with gr.Column(scale=1):
                label_output = gr.HTML(label="Predicted Emotion")
                scores_output = gr.Markdown(label="Class Scores")
                reasoning_output = gr.Textbox(
                    label="LLM Reasoning", lines=6, interactive=False
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

def _build_realtime_demo(cfg, args):
    """Gradio Blocks app: live sliding-window predictions from webcam/video."""
    import gradio as gr

    from vie_gameemo.inference.realtime import RealtimeInferenceRunner

    runner_state: dict = {"runner": None, "history": []}

    def start_inference(video_path: str | None) -> str:
        if video_path is None:
            return "Please upload a video file."
        runner_state["history"] = []
        runner = RealtimeInferenceRunner(
            checkpoint=args.checkpoint,
            cfg=cfg,
            window_seconds=5.0,
            step_seconds=1.0,
            max_latency_ms=600.0,
            skip_llm=True,
        )
        runner_state["runner"] = runner

        def on_pred(pred):
            runner_state["history"].append(pred)

        import threading
        t = threading.Thread(
            target=runner.process_stream,
            kwargs={"source": Path(video_path), "on_prediction": on_pred},
            daemon=True,
        )
        t.start()
        runner_state["thread"] = t
        return "Inference started..."

    def get_history() -> str:
        history = runner_state.get("history", [])
        if not history:
            return "_No predictions yet..._"
        lines = []
        for item in history[-10:]:
            label = item.get("label", "?")
            conf = item.get("confidence", 0)
            t_start = item.get("start_sec", 0)
            t_end = item.get("end_sec", 0)
            emoji = _EMOTION_EMOJI.get(label, "❓")
            lines.append(
                f"[{t_start:.1f}s–{t_end:.1f}s] {emoji} **{label}** ({conf:.1%})"
            )
        return "\n".join(lines)

    with gr.Blocks(title="Vie-GameEmo — Real-time Demo") as demo:
        gr.Markdown("# Vie-GameEmo Real-time Demo\nSliding-window emotion tracking.")
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload video (or use webcam)")
                start_btn = gr.Button("Start", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                history_md = gr.Markdown(label="Prediction history (last 10 windows)")
                refresh_btn = gr.Button("Refresh")

        start_btn.click(fn=start_inference, inputs=[video_input], outputs=[status])
        refresh_btn.click(fn=get_history, inputs=[], outputs=[history_md])

    return demo


if __name__ == "__main__":
    sys.exit(main())
