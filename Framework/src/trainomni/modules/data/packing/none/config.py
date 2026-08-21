"""No-packing policy configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoPackingConfig:
    pass
