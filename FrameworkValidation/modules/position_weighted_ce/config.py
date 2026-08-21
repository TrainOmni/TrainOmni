"""Configuration for the validation-only position-weighted CE objective."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionWeightedCEConfig:
    """Weight later causal targets more heavily than earlier targets."""

    ignore_index: int = -100
    label_smoothing: float = 0.0
    final_token_weight: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.final_token_weight <= 0.0:
            raise ValueError("final_token_weight must be positive")
