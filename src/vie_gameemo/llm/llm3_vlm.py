"""LLM-3: VLM End-to-End with LoRA fine-tune.

Uses Qwen2.5-VL as the backbone. Receives raw video frames + optional
spectrogram image (audio-as-image trick), predicts emotion + reasoning
in one forward pass. Fine-tuned via LoRA on Vie-GameEmo annotated data.
"""

import logging
from pathlib import Path

from vie_gameemo.llm.base import BaseLLMReasoner, LLMOutput
from vie_gameemo.llm.llm1_explainer import LLM1Explainer, _make_bnb_config

logger = logging.getLogger(__name__)

_VLM_SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích cảm xúc game streaming. Nhìn vào các hình ảnh từ clip sau và cho biết cảm xúc của streamer.

Trả lời theo format CHÍNH XÁC:
<think>
[Phân tích 3-5 câu về biểu cảm, âm thanh, và bối cảnh game]
</think>
<answer>[neutral/hype/amused/tilted/sad/shocked/fear/disgusted]</answer>
"""


class LLM3VLMEndToEnd(BaseLLMReasoner):
    """VLM end-to-end reasoner with LoRA fine-tune."""

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        n_input_frames: int = 8,
        include_spectrogram_image: bool = True,
        quantization: str = "4bit",
        max_new_tokens: int = 400,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        adapter_path: Path | None = None,
    ) -> None:
        """Initialize.

        Args:
            vlm_model: HF VLM model ID.
            n_input_frames: Frames sampled from clip as VLM input.
            include_spectrogram_image: Include spectrogram as additional image.
            quantization: '4bit' | '8bit' | 'none'.
            max_new_tokens: Max gen length.
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
            lora_target_modules: Modules to apply LoRA to.
            adapter_path: Path to pretrained LoRA adapter.
        """
        self.vlm_model = vlm_model
        self.n_input_frames = n_input_frames
        self.include_spectrogram_image = include_spectrogram_image
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load VLM + apply LoRA adapter if available."""
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as QVLM

        logger.info("Loading LLM-3 VLM: %s (quant=%s)", self.vlm_model, self.quantization)
        bnb_cfg = _make_bnb_config(self.quantization)
        kwargs: dict = {"device_map": "auto"}
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = QVLM.from_pretrained(self.vlm_model, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.vlm_model)

        if self.adapter_path is not None and Path(self.adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
            logger.info("Loaded LoRA adapter from %s", self.adapter_path)

        self.model.eval()
        logger.info("LLM-3 VLM loaded")

    def unload(self) -> None:
        """Free VRAM."""
        import gc
        import torch
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reason(self, evidence: dict) -> LLMOutput:
        """Reason on a single clip using raw images.

        Args:
            evidence: Dict with keys:
                'frame_paths': list of frame Path objects,
                'audio_path': Path to wav (for spectrogram),
                'transcript': str.

        Returns:
            LLMOutput.
        """
        if self.model is None:
            self.load()

        import numpy as np
        import torch
        from PIL import Image

        frame_paths: list[Path] = evidence.get("frame_paths", [])
        audio_path: Path | None = evidence.get("audio_path")
        transcript: str = evidence.get("transcript", "")

        # Sample frames
        if frame_paths:
            indices = np.linspace(0, len(frame_paths) - 1, self.n_input_frames, dtype=int)
            frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]
        else:
            frames = []

        # Optional spectrogram
        if self.include_spectrogram_image and audio_path and Path(audio_path).exists():
            try:
                spec_img = self.render_spectrogram_image(Path(audio_path))
                frames.append(spec_img)
            except Exception as exc:
                logger.debug("Spectrogram render failed: %s", exc)

        images = frames if frames else None
        text_prompt = _VLM_SYSTEM_PROMPT
        if transcript:
            text_prompt += f'\nTranscript: "{transcript}"'

        content = []
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": "user", "content": content}]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            images=images if images else None,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )

        out = out_ids[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        reasoning, answer, fmt_valid = LLM1Explainer.parse_output(raw)
        return LLMOutput(reasoning=reasoning, answer=answer, raw=raw, format_valid=fmt_valid)

    def reason_batch(self, evidences: list[dict]) -> list[LLMOutput]:
        """Batch reason (sequential).

        Args:
            evidences: List of evidence dicts.

        Returns:
            List of LLMOutput.
        """
        return [self.reason(e) for e in evidences]

    @staticmethod
    def render_spectrogram_image(audio_path: Path) -> "Image.Image":
        """Render audio spectrogram as RGB image for VLM input.

        Args:
            audio_path: Path to wav file.

        Returns:
            PIL RGB Image of the spectrogram.
        """
        import io

        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=56)  # 224x224 pixels
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
