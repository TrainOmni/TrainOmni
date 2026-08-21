"""Configuration for complete full-parameter training."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FullParameterConfig:
    group_name: str = "all"
    group_by_top_level_component: bool = False

    def __post_init__(self) -> None:
        if not self.group_name.strip():
            raise ValueError("group_name must not be empty")
        if not isinstance(self.group_by_top_level_component, bool):
            raise TypeError("group_by_top_level_component must be a boolean")
