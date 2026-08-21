"""Selected-component full-parameter configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentParameterConfig:
    train: tuple[str, ...]
    group_per_component: bool = True

    def __post_init__(self) -> None:
        if not self.train or any(not name.strip() for name in self.train):
            raise ValueError("train must contain non-empty component names")
