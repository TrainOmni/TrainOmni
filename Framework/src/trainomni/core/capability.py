"""Capability declarations used during module preflight."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .errors import CapabilityError


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """Immutable set with deterministic diagnostics."""

    values: frozenset[str] = frozenset()

    @classmethod
    def of(cls, values: Iterable[str] = ()) -> CapabilitySet:
        normalized = frozenset(value.strip() for value in values if value.strip())
        return cls(normalized)

    def union(self, *others: CapabilitySet) -> CapabilitySet:
        values = set(self.values)
        for other in others:
            values.update(other.values)
        return CapabilitySet(frozenset(values))

    def require(self, required: CapabilitySet, *, owner: str) -> None:
        missing = sorted(required.values - self.values)
        if missing:
            raise CapabilityError(f"{owner} requires missing capabilities: {', '.join(missing)}")
