"""Causal language-model objective configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True, kw_only=True)
class CausalLMConfig:
    ignore_index: int = -100
    reduction: Literal["token_mean", "sample_mean"] = "token_mean"
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.reduction not in {"token_mean", "sample_mean"}:
            raise ValueError(f"unsupported reduction: {self.reduction}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
