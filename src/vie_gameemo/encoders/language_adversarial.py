"""Language-adversarial head with Gradient Reversal Layer (GRL).

Forces h_text to NOT encode language information, reducing embedding
fragmentation (R1) and blocking language→class confound shortcuts (R2).

Architecture:
    h_text → [GRL(lambda)] → MLP(D→128→2) → language prediction (vi/en)

The GRL reverses gradients during backprop: the encoder learns to produce
embeddings that FOOL the language discriminator.

Loss contribution:
    L_total = L_emotion + lambda_grl * L_lang_adv

Toggle via config: text_encoder.language_adversarial.enabled
Default: OFF (to preserve baseline comparability).
"""

import torch
from torch import Tensor, nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x: Tensor, lambda_grl: float) -> Tensor:
        ctx.lambda_grl = lambda_grl
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps _GradientReversalFunction as an nn.Module."""

    def __init__(self, lambda_grl: float = 1.0) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x: Tensor) -> Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_grl)

    def set_lambda(self, value: float) -> None:
        self.lambda_grl = value


class LanguageDiscriminator(nn.Module):
    """Small MLP that predicts source_language from h_text.

    Connected to the text encoder via GRL — encoder is trained to
    make this discriminator FAIL (language-invariant embeddings).

    Args:
        d_input: Dimension of h_text (e.g., 768 for XLM-R base, 1024 for large).
        d_hidden: Hidden layer dimension.
        lambda_grl: GRL scaling factor. Higher = stronger adversarial signal.
    """

    def __init__(
        self,
        d_input: int = 768,
        d_hidden: int = 128,
        lambda_grl: float = 0.1,
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(lambda_grl)
        self.classifier = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 2),
        )
        self._lang_to_idx = {"vi": 0, "en": 1}

    def forward(self, h_text: Tensor) -> Tensor:
        """Forward pass through GRL + classifier.

        Args:
            h_text: (B, T, D) or (B, D) text embeddings.

        Returns:
            (B, 2) language logits.
        """
        if h_text.dim() == 3:
            h_text = h_text.mean(dim=1)
        reversed_h = self.grl(h_text)
        return self.classifier(reversed_h)

    def compute_loss(
        self,
        h_text: Tensor,
        source_languages: list[str],
    ) -> Tensor:
        """Compute cross-entropy loss for language prediction.

        Args:
            h_text: (B, T, D) or (B, D) text embeddings.
            source_languages: List of "vi"/"en" per sample.

        Returns:
            Scalar loss.
        """
        logits = self.forward(h_text)
        targets = torch.tensor(
            [self._lang_to_idx.get(l, 0) for l in source_languages],
            dtype=torch.long,
            device=logits.device,
        )
        return nn.functional.cross_entropy(logits, targets)

    def language_accuracy(
        self,
        h_text: Tensor,
        source_languages: list[str],
    ) -> float:
        """Compute language classification accuracy (for monitoring).

        When adversarial training works, this should drop toward 50% (random).
        """
        with torch.no_grad():
            logits = self.forward(h_text)
            preds = logits.argmax(dim=-1)
            targets = torch.tensor(
                [self._lang_to_idx.get(l, 0) for l in source_languages],
                dtype=torch.long,
                device=logits.device,
            )
            return float((preds == targets).float().mean().item())

    def set_lambda(self, value: float) -> None:
        self.grl.set_lambda(value)


def build_language_adversarial(encoder_cfg) -> LanguageDiscriminator | None:
    """Build LanguageDiscriminator from config if enabled.

    Args:
        encoder_cfg: text_encoder config namespace.

    Returns:
        LanguageDiscriminator or None if disabled.
    """
    adv_cfg = getattr(encoder_cfg, "language_adversarial", None)
    if adv_cfg is None or not getattr(adv_cfg, "enabled", False):
        return None

    lambda_grl = getattr(adv_cfg, "lambda_grl", 0.1)

    # Infer d_input from encoder backend
    backend = getattr(encoder_cfg, "backend", "cafebert")
    d_input = 1024 if backend in ("cafebert", "xlmr-large", "mE5-frozen") else 768

    return LanguageDiscriminator(d_input=d_input, lambda_grl=lambda_grl)
