"""Storage-neutral records emitted before semantic sample adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ._identity import freeze_mapping, normalize_identity


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
        object.__setattr__(
            self,
            "sample_id",
            normalize_identity(self.sample_id, field="data record sample_id"),
        )
        object.__setattr__(
            self,
            "source",
            normalize_identity(self.source, field="data record source"),
        )
        object.__setattr__(
            self, "fields", freeze_mapping(self.fields, field="data record fields")
        )
        object.__setattr__(
            self,
            "position",
            freeze_mapping(self.position, field="data record position"),
        )
