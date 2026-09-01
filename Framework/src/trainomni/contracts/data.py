"""Storage-neutral records emitted before semantic sample adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class DataRecord:
    """One physical row plus stable source coordinates.

    Data sources own physical I/O and cursors.  A data-adapter module owns the
    conversion from ``fields`` into an :class:`OmniSample`.
    """

    sample_id: str
    fields: Mapping[str, Any]
    source: str
    position: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("data record sample_id must not be empty")
        if not self.source:
            raise ValueError("data record source must not be empty")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "position", MappingProxyType(dict(self.position)))
