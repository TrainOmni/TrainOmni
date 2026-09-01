"""Assemble the canonical data path into a resumable batch stream."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from trainomni.core.context import BuildContext
from trainomni.core.errors import CheckpointError
from trainomni.core.module import ModuleKind
from trainomni.core.resolver import ModuleResolver
from trainomni.modules.data.adapters.binding import AdaptedSource
from trainomni.runtime.execution.data import RankShardedSource
from trainomni.specs.task import DataPipelineSpec


def _optional_state(module):
    hook = getattr(module, "state_dict", None)
    return None if not callable(hook) else dict(hook())


def _restore_optional_state(module, state, *, owner: str) -> None:
    hook = getattr(module, "load_state_dict", None)
    if state is None:
        if callable(hook):
            hook({})
        return
    if not callable(hook):
        raise CheckpointError(f"{owner} checkpoint state has no load_state_dict owner")
    hook(state)


class DataPipelineStream:
    def __init__(
        self,
        *,
        source,
        transforms,
        model_io,
        supervision,
        packer,
        collator,
    ) -> None:
        self.source = source
        self.transforms = tuple(transforms)
        self.model_io = model_io
        self.supervision = supervision
        self.packer = packer
        self.collator = collator
        self._ready = []

    def shard(
        self,
        *,
        rank: int,
        world_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> None:
        if world_size == 1 and num_workers == 1:
            return
        if isinstance(self.source, RankShardedSource):
            if (self.source.rank, self.source.world_size) != (rank, world_size):
                raise CheckpointError("data stream was already sharded for another topology")
            return
        if self._ready:
            raise CheckpointError("data stream must be sharded before reading samples")
        physical_shard = getattr(self.source, "shard", None)
        if callable(physical_shard):
            physical_shard(
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            return
        if num_workers != 1:
            raise CheckpointError(
                "generic sample sources do not support physical worker sharding"
            )
        self.source = RankShardedSource(
            self.source,
            rank=rank,
            world_size=world_size,
        )

    def next_batch(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        while len(self._ready) < batch_size:
            sample = self.source.next_sample()
            for transform in self.transforms:
                sample = transform.apply(sample)
            encoded = self.model_io.encode(sample)
            supervised = self.supervision.annotate(encoded)
            ready = self.packer.add(supervised)
            if ready:
                self._ready.extend(ready)
        examples = tuple(self._ready[:batch_size])
        del self._ready[:batch_size]
        return self.collator.collate(examples)

    def state_dict(self):
        return {
            "source": dict(self.source.state_dict()),
            "packer": dict(self.packer.state_dict()),
            "transforms": tuple(_optional_state(item) for item in self.transforms),
            "model_io": _optional_state(self.model_io),
            "supervision": _optional_state(self.supervision),
            "collator": _optional_state(self.collator),
            "ready": tuple(self._ready),
        }

    def metrics(self):
        hook = getattr(self.source, "metrics", None)
        return {} if not callable(hook) else dict(hook())

    def load_state_dict(self, state):
        expected = {
            "source",
            "packer",
            "transforms",
            "model_io",
            "supervision",
            "collator",
            "ready",
        }
        if set(state) != expected:
            raise CheckpointError(
                "data pipeline checkpoint state is incomplete or has unknown fields"
            )
        ready = state["ready"]
        if not isinstance(ready, (tuple, list)):
            raise CheckpointError("data pipeline ready buffer must be a sequence")
        transform_states = state["transforms"]
        if not isinstance(transform_states, (tuple, list)) or len(
            transform_states
        ) != len(self.transforms):
            raise CheckpointError("data transform state count mismatch")
        self.source.load_state_dict(state["source"])
        self.packer.load_state_dict(state["packer"])
        for index, (transform, transform_state) in enumerate(
            zip(self.transforms, transform_states, strict=True)
        ):
            _restore_optional_state(
                transform, transform_state, owner=f"transform[{index}]"
            )
        _restore_optional_state(self.model_io, state["model_io"], owner="model_io")
        _restore_optional_state(
            self.supervision, state["supervision"], owner="supervision"
        )
        _restore_optional_state(self.collator, state["collator"], owner="collator")
        self._ready = list(ready)


def build_data_stream(
    spec: DataPipelineSpec,
    resolver: ModuleResolver,
    *,
    context: BuildContext,
) -> DataPipelineStream:
    child_sources = {
        name: resolver.resolve(reference, kind=ModuleKind.DATA_SOURCE).build(context)
        for name, reference in spec.sources
    }
    source_context = replace(
        context,
        components=MappingProxyType(child_sources),
    )
    source = resolver.resolve(spec.source, kind=ModuleKind.DATA_SOURCE).build(
        source_context
    )
    if spec.adapter is not None:
        adapter = resolver.resolve(
            spec.adapter,
            kind=ModuleKind.DATA_ADAPTER,
        ).build(context)
        source = AdaptedSource(source, adapter)
    transforms = tuple(
        resolver.resolve(reference, kind=ModuleKind.SAMPLE_TRANSFORM).build(context)
        for reference in spec.transforms
    )
    model_io = resolver.resolve(spec.model_io, kind=ModuleKind.MODEL_IO).build(context)
    supervision = resolver.resolve(
        spec.supervision, kind=ModuleKind.SUPERVISION
    ).build(context)
    packer = resolver.resolve(spec.packer, kind=ModuleKind.PACKER).build(context)
    collator = resolver.resolve(spec.collator, kind=ModuleKind.COLLATOR).build(context)
    return DataPipelineStream(
        source=source,
        transforms=transforms,
        model_io=model_io,
        supervision=supervision,
        packer=packer,
        collator=collator,
    )
