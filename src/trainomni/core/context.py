"""Narrow construction and objective contexts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class BuildContext:
    task_digest: str
    task_root: Path | None = None
    components: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class ObjectiveContext:
    global_step: int
    micro_step: int
    training: bool = True
