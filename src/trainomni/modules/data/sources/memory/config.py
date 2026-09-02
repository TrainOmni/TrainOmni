"""In-memory canonical fixture source configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trainomni.modules.data._validation import require_bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorySourceConfig:
    samples: tuple[Mapping[str, Any], ...]
    repeat: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.samples, (tuple, list)) or not self.samples:
            raise ValueError("memory source requires at least one sample")
        if any(not isinstance(sample, Mapping) for sample in self.samples):
            raise TypeError("memory source samples must be mappings")
        require_bool(self.repeat, field="memory source repeat")
        object.__setattr__(self, "samples", tuple(self.samples))
