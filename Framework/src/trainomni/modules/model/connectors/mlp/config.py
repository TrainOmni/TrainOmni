"""MLP connector configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class MLPConnectorConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: str = "gelu"
    bias: bool = True
    layer_norm: bool = False

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.output_dim) <= 0:
            raise ValueError("connector dimensions must be positive")
        if self.activation not in {"gelu", "silu", "relu"}:
            raise ValueError("activation must be gelu, silu, or relu")
