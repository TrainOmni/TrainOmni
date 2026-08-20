"""Process-wide deterministic seeding before model construction."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any


def seed_everything(seed: int, engine_config: Mapping[str, Any]) -> None:
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if engine_config.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
