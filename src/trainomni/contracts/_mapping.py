"""Small pickle-safe immutable mapping used by process-bound contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class FrozenDict(Mapping[K, V], Generic[K, V]):
    __slots__ = ("_data",)

    def __init__(self, value: Mapping[K, V] | None = None) -> None:
        if value is not None and not isinstance(value, Mapping):
            raise TypeError("FrozenDict value must be a mapping")
        self._data = dict(value or {})

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __reduce__(self):
        return (type(self), (self._data,))
