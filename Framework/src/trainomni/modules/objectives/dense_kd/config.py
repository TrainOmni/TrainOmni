"""Offline dense-logit distillation configuration."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseKDConfig:
    ce_weight: float = 0.5
    kd_weight: float = 0.5
    temperature: float = 2.0
    teacher_logits_field: str = "teacher_logits"
    ignore_index: int = -100
    reduction: str = "token_mean"

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.ce_weight, self.kd_weight, self.temperature)
        ):
            raise ValueError("KD weights and temperature must be finite")
        if self.ce_weight < 0 or self.kd_weight < 0:
            raise ValueError("KD loss weights must be non-negative")
        if self.ce_weight + self.kd_weight <= 0:
            raise ValueError("at least one KD loss weight must be positive")
        if self.temperature <= 0:
            raise ValueError("KD temperature must be positive")
        if self.reduction not in {"token_mean", "sample_mean"}:
            raise ValueError("reduction must be token_mean or sample_mean")
        if not self.teacher_logits_field:
            raise ValueError("teacher_logits_field must not be empty")
