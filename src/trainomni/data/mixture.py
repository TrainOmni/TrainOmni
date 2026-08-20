"""Deterministic weighted dataset mixing with exact restart state."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from typing import Any

from .importers import ImportedSample
from .stream import DatasetStream

MIXTURE_STATE_VERSION = "trainomni.mixture-state.v1"


class MixtureError(ValueError):
    pass


class MixtureStream:
    """Draw from weighted streams without hiding reader progress.

    Each accepted draw advances exactly one physical reader. With ``repeat=True``
    a depleted non-empty stream starts a new epoch. The RNG, active set, per-stream
    reader state, and epoch counters are all checkpointed.
    """

    def __init__(
        self,
        streams: tuple[DatasetStream, ...],
        *,
        seed: int,
        repeat: bool,
    ) -> None:
        if not streams:
            raise MixtureError("dataset mixture requires at least one stream")
        ids = [stream.spec.dataset_id for stream in streams]
        if len(set(ids)) != len(ids):
            raise MixtureError("dataset mixture IDs must be unique")
        self._streams = {stream.spec.dataset_id: stream for stream in streams}
        self._order = tuple(ids)
        self._weights = {
            stream.spec.dataset_id: float(stream.spec.weight) for stream in streams
        }
        self._rng = random.Random(seed)
        self._repeat = repeat
        self._active = set(ids)
        self._epochs = {dataset_id: 0 for dataset_id in ids}
        self._draw_count = 0
        self._iterators: dict[str, Iterator[ImportedSample]] = {}

    def __iter__(self) -> MixtureStream:
        return self

    def __next__(self) -> ImportedSample:
        while self._active:
            dataset_id = self._choose()
            iterator = self._iterators.setdefault(
                dataset_id, iter(self._streams[dataset_id])
            )
            try:
                item = next(iterator)
            except StopIteration:
                self._iterators.pop(dataset_id, None)
                if not self._repeat:
                    self._active.remove(dataset_id)
                    continue
                self._streams[dataset_id].reset()
                self._epochs[dataset_id] += 1
                iterator = iter(self._streams[dataset_id])
                self._iterators[dataset_id] = iterator
                try:
                    item = next(iterator)
                except StopIteration:
                    # Empty sources cannot be repeated and must not spin forever.
                    self._active.remove(dataset_id)
                    self._iterators.pop(dataset_id, None)
                    continue
            self._draw_count += 1
            return item
        raise StopIteration

    def _choose(self) -> str:
        candidates = [item for item in self._order if item in self._active]
        total = sum(self._weights[item] for item in candidates)
        point = self._rng.random() * total
        cumulative = 0.0
        for dataset_id in candidates:
            cumulative += self._weights[dataset_id]
            if point < cumulative:
                return dataset_id
        return candidates[-1]

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": MIXTURE_STATE_VERSION,
            "order": self._order,
            "weights": dict(self._weights),
            "repeat": self._repeat,
            "active": tuple(item for item in self._order if item in self._active),
            "epochs": dict(self._epochs),
            "draw_count": self._draw_count,
            "rng_state": self._rng.getstate(),
            "streams": {
                dataset_id: self._streams[dataset_id].state_dict()
                for dataset_id in self._order
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("state_version") != MIXTURE_STATE_VERSION:
            raise MixtureError("mixture state version mismatch")
        if tuple(state.get("order", ())) != self._order:
            raise MixtureError("mixture dataset order mismatch")
        if state.get("weights") != self._weights or state.get("repeat") != self._repeat:
            raise MixtureError("mixture weights/repeat policy mismatch")
        streams = state.get("streams")
        active = state.get("active")
        epochs = state.get("epochs")
        draw_count = state.get("draw_count")
        if not isinstance(streams, Mapping) or set(streams) != set(self._order):
            raise MixtureError("mixture child stream state mismatch")
        if not isinstance(active, (tuple, list)) or not set(active) <= set(self._order):
            raise MixtureError("mixture active set is invalid")
        if not isinstance(epochs, Mapping) or set(epochs) != set(self._order):
            raise MixtureError("mixture epoch state mismatch")
        if not isinstance(draw_count, int) or draw_count < 0:
            raise MixtureError("mixture draw_count is invalid")
        for dataset_id in self._order:
            child = streams[dataset_id]
            if not isinstance(child, Mapping):
                raise MixtureError("mixture child state must be a mapping")
            self._streams[dataset_id].load_state_dict(child)
        try:
            self._rng.setstate(state["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MixtureError("mixture RNG state is invalid") from exc
        self._active = set(active)
        self._epochs = {item: int(epochs[item]) for item in self._order}
        self._draw_count = draw_count
        self._iterators.clear()

    @property
    def draw_count(self) -> int:
        return self._draw_count
