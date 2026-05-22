"""Reproducibility: seed everything for deterministic experiments.

Non-determinism note: CUDA reductions (e.g., atomicAdd) may still be
non-deterministic on some GPU/driver combinations even with these settings.
Set CUBLAS_WORKSPACE_CONFIG=:4096:8 env var for stricter determinism at
the cost of memory.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed all RNG sources for reproducibility.

    Seeds: Python `random`, NumPy, PyTorch (CPU + CUDA), and sets
    `cudnn.deterministic = True`, `cudnn.benchmark = False`.

    Args:
        seed: Seed value. Must be a non-negative integer.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d", seed)
