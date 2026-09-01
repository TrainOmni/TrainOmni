"""In-memory canonical fixture source configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorySourceConfig:
    samples: tuple[Mapping[str, Any], ...]
    repeat: bool = True

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("memory source requires at least one sample")
