"""Linear connector configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearConnectorConfig:
    input_dim: int
    output_dim: int
    bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("connector dimensions must be positive")
