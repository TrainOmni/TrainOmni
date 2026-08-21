"""Freeze-list parameter configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class FreezeParameterConfig:
    freeze: tuple[str, ...]
    group_name: str = "unfrozen"

    def __post_init__(self) -> None:
        if not self.freeze or any(not name.strip() for name in self.freeze):
            raise ValueError("freeze must contain non-empty component names")
        if not self.group_name.strip():
            raise ValueError("group_name must not be empty")
