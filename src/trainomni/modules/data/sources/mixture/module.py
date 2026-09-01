"""Counter-based deterministic weighted sampling over stateful sources."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.specs.digest import identity_digest

from .config import MixtureSourceConfig


class MixtureSource:
    def __init__(self, config: MixtureSourceConfig, sources: Mapping[str, Any]) -> None:
        if set(sources) != set(config.weights):
            raise SpecError(
                "mixture weights and data.sources names differ: "
                f"weights={sorted(config.weights)}, sources={sorted(sources)}"
            )
        for name, source in sources.items():
            for hook in ("next_sample", "state_dict", "load_state_dict"):
                if not callable(getattr(source, hook, None)):
                    raise SpecError(f"mixture child {name!r} has no callable {hook}")
        self.config = config
        self.sources = dict(sorted(sources.items()))
        self.active_sources = tuple(self.sources)
        self.cursor = 0
        self.counts = {name: 0 for name in self.sources}
        self.identity = identity_digest(
            {
                "schema_version": 1,
                "seed": config.seed,
                "weights": dict(config.weights),
                "namespace_sample_ids": config.namespace_sample_ids,
            }
        )

    def _source_name(self) -> str:
        total = sum(self.config.weights[name] for name in self.active_sources)
        cumulative = 0.0
        thresholds = []
        for name in self.active_sources:
            cumulative += self.config.weights[name] / total
            thresholds.append((cumulative, name))
        thresholds[-1] = (1.0, thresholds[-1][1])
        payload = f"{self.config.seed}:{self.cursor}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        for threshold, name in thresholds:
            if value < threshold:
                return name
        raise RuntimeError("mixture threshold construction is invalid")

    def next_sample(self):
        while self.active_sources:
            name = self._source_name()
            try:
                sample = self.sources[name].next_sample()
            except StopIteration:
                self.active_sources = tuple(
                    candidate
                    for candidate in self.active_sources
                    if candidate != name
                )
                continue
            break
        else:
            raise StopIteration
        self.cursor += 1
        self.counts[name] += 1
        metadata = dict(sample.metadata)
        if "trainomni.source" in metadata:
            raise SpecError(
                "sample metadata key 'trainomni.source' is reserved by mixture source"
            )
        metadata["trainomni.source"] = name
        sample_id = (
            f"{name}::{sample.sample_id}"
            if self.config.namespace_sample_ids
            else sample.sample_id
        )
        return replace(sample, sample_id=sample_id, metadata=metadata)

    def state_dict(self):
        return {
            "schema_version": 1,
            "identity": self.identity,
            "cursor": self.cursor,
            "counts": dict(self.counts),
            "active_sources": self.active_sources,
            "sources": {
                name: dict(source.state_dict())
                for name, source in self.sources.items()
            },
        }

    def metrics(self):
        return {
            "data/mixture/samples": self.cursor,
            "data/mixture/active_sources": len(self.active_sources),
            **{
                f"data/source/{name}/samples": count
                for name, count in self.counts.items()
            },
        }

    def load_state_dict(self, state):
        expected = {
            "schema_version",
            "identity",
            "cursor",
            "counts",
            "active_sources",
            "sources",
        }
        if set(state) != expected or state["schema_version"] != 1:
            raise CheckpointError("invalid mixture source state")
        if state["identity"] != self.identity:
            raise CheckpointError("mixture source identity changed")
        counts = state["counts"]
        active_sources = state["active_sources"]
        sources = state["sources"]
        if not isinstance(counts, Mapping) or set(counts) != set(self.sources):
            raise CheckpointError("mixture source count names changed")
        if not isinstance(sources, Mapping) or set(sources) != set(self.sources):
            raise CheckpointError("mixture child source names changed")
        if not isinstance(active_sources, (tuple, list)):
            raise CheckpointError("mixture active source state is not a sequence")
        normalized_active = tuple(str(name) for name in active_sources)
        if len(normalized_active) != len(set(normalized_active)) or not set(
            normalized_active
        ).issubset(self.sources):
            raise CheckpointError("mixture active source names are invalid")
        cursor = int(state["cursor"])
        normalized_counts = {name: int(counts[name]) for name in self.sources}
        if cursor < 0 or any(value < 0 for value in normalized_counts.values()):
            raise CheckpointError("mixture cursor and counts must be non-negative")
        if sum(normalized_counts.values()) != cursor:
            raise CheckpointError("mixture source counts do not sum to cursor")
        for name, source in self.sources.items():
            source.load_state_dict(sources[name])
        self.cursor = cursor
        self.counts = normalized_counts
        self.active_sources = normalized_active


def _factory(config: MixtureSourceConfig, context):
    return MixtureSource(config, context.components)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("data_source:trainomni/mixture@1"),
        config_type=MixtureSourceConfig,
        factory=_factory,
        provides=CapabilitySet.of({"data.sample.omni", "data.source.stateful"}),
        requires=CapabilitySet.of({"data.sample.omni", "data.source.stateful"}),
    )
