"""Deterministic process RNG initialization."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)
