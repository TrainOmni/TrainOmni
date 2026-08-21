"""Minimal native LoRA configuration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class LoRAParameterConfig:
    target_patterns: tuple[str, ...]
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    train_bias: bool = False
    group_name: str = "lora"

    def __post_init__(self) -> None:
        if not self.target_patterns:
            raise ValueError("LoRA requires at least one target pattern")
        for pattern in self.target_patterns:
            re.compile(pattern)
        if self.rank <= 0 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.group_name:
            raise ValueError("LoRA group_name must not be empty")
