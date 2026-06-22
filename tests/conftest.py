"""Shared pytest fixtures for Vie-GameEmo tests."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BATCH_SIZE = 2
D_MODEL = 768
SEQ_LEN_AUDIO = 64
SEQ_LEN_FACE = 16
SEQ_LEN_CTX = 16
SEQ_LEN_TEXT = 1
N_CLASSES = 8


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_tensors():
    """Minimal batch of modality tensors for fusion/classifier tests."""
    return {
        "audio": torch.randn(BATCH_SIZE, SEQ_LEN_AUDIO, D_MODEL),
        "face": torch.randn(BATCH_SIZE, SEQ_LEN_FACE, D_MODEL),
        "context": torch.randn(BATCH_SIZE, SEQ_LEN_CTX, D_MODEL),
        "text": torch.randn(BATCH_SIZE, SEQ_LEN_TEXT, D_MODEL),
        "has_face": torch.ones(BATCH_SIZE, dtype=torch.bool),
        "label": torch.randint(0, N_CLASSES, (BATCH_SIZE,)),
    }


@pytest.fixture
def batch_tensors_no_face(batch_tensors):
    """Batch where has_face=False for all samples."""
    batch = dict(batch_tensors)
    batch["has_face"] = torch.zeros(BATCH_SIZE, dtype=torch.bool)
    return batch
