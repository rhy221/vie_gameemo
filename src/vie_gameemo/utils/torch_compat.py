"""PyTorch version-compatibility shims.

Call `ensure_set_submodule_patch()` before loading any quantized (bitsandbytes
4-bit/8-bit) HF model — quantizer code calls `nn.Module.set_submodule`, which
only exists on PyTorch >= 2.1. Older PyTorch raises
`AttributeError: '...' object has no attribute 'set_submodule'` deep inside
`transformers.integrations.bitsandbytes.replace_with_bnb_linear`.
"""

import torch.nn as nn


def ensure_set_submodule_patch() -> None:
    """Monkey-patch `nn.Module.set_submodule` if missing (PyTorch < 2.1)."""
    if hasattr(nn.Module, "set_submodule"):
        return

    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.split(".")
        parent = self
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    nn.Module.set_submodule = _set_submodule  # type: ignore[method-assign]
