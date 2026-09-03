"""Assemble the canonical data path into a resumable batch stream."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from torch.utils.data import IterableDataset, get_worker_info

from trainomni.contracts.batch import SupervisedExample
from trainomni.core.context import BuildContext
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.core.module import ModuleKind
from trainomni.core.resolver import ModuleResolver
from trainomni.modules.data.adapters.binding import AdaptedSource
from trainomni.runtime.execution.data import RankShardedSource
from trainomni.specs.task import DataPipelineSpec


def _restore_worker_pipeline(local_sources, task_root, payload):
    # Spawn must import pinned task-local classes before unpickling their instances.
    # This is the same explicitly trusted code as the parent's local-module opt-in.
    from trainomni.catalog.local import load_local_descriptor

    for source in local_sources:
        load_local_descriptor(source, task_root=Path(task_root), allow_local_code=True)
    result = DataPipelineStream.__new__(DataPipelineStream)
    result.__dict__.update(pickle.loads(payload))
    return result


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


class DataPipelineStream(IterableDataset):
    def __init__(
        self,
        *,
        source,
        transforms,
        model_io,
        supervision,
        packer,
        collator,
        drop_last: bool,
    ) -> None:
        self.source = source
        self.transforms = tuple(transforms)
        self.model_io = model_io
        self.supervision = supervision
        self.packer = packer
        self.collator = collator
        self.drop_last = drop_last
        self._ready = []
        self._exhausted = False
        self._dropped_examples = 0
        self._loader_batch_size: int | None = None
        self._loader_rank = 0
        self._loader_world_size = 1
        self._worker_topology: tuple[int, int] | None = None
        self._local_sources = ()
        self._task_root = None

    def bind_local_sources(self, sources, task_root) -> None:
        self._local_sources = tuple(sources)
        self._task_root = None if task_root is None else str(task_root)

    def __reduce__(self):
        return (
            _restore_worker_pipeline,
            (self._local_sources, self._task_root, pickle.dumps(self.__dict__)),
        )

    def configure_loader(
        self,
        *,
        batch_size: int,
        rank: int,
        world_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("loader rank must be in range")
        if self._ready or self._exhausted:
            raise CheckpointError("data loader must be configured before reading samples")
        self._loader_batch_size = batch_size
        self._loader_rank = rank
        self._loader_world_size = world_size

    def __iter__(self):
        if self._loader_batch_size is None:
            raise RuntimeError("data pipeline has not been configured for a loader")
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        topology = (worker_id, num_workers)
        if self._worker_topology is None:
            self.shard(
                rank=self._loader_rank,
                world_size=self._loader_world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            self._worker_topology = topology
        elif self._worker_topology != topology:
            raise CheckpointError("data pipeline worker topology changed")
        while True:
            try:
                yield self.next_batch(self._loader_batch_size)
            except StopIteration:
                return

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
        if world_size > 1 and getattr(self.source, "is_finite", None) is not False:
            raise SpecError(
                "multi-rank training requires an explicitly repeating source; "
                "finite or unknown exhaustion cannot guarantee equal optimizer steps"
            )
        if isinstance(self.source, RankShardedSource):
            if (self.source.rank, self.source.world_size) != (rank, world_size):
                raise CheckpointError("data stream was already sharded for another topology")
            return
        if self._ready or self._exhausted:
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
        while len(self._ready) < batch_size and not self._exhausted:
            try:
                sample = self.source.next_sample()
            except StopIteration:
                self._exhausted = True
                flush = getattr(self.packer, "flush", None)
                if not callable(flush):
                    raise SpecError(
                        "finite data requires a packer with an explicit flush() contract"
                    ) from None
                self._ready.extend(flush())
                break
            for transform in self.transforms:
                sample = transform.apply(sample)
            encoded = self.model_io.encode(sample)
            supervised = self.supervision.annotate(encoded)
            ready = self.packer.add(supervised)
            if ready:
                self._ready.extend(ready)
        if len(self._ready) < batch_size:
            if not self._ready:
                raise StopIteration
            if self.drop_last:
                self._dropped_examples += len(self._ready)
                self._ready.clear()
                raise StopIteration
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
            "exhausted": self._exhausted,
            "dropped_examples": self._dropped_examples,
            "worker_topology": self._worker_topology,
        }

    def metrics(self):
        hook = getattr(self.source, "metrics", None)
        values = {} if not callable(hook) else dict(hook())
        values["data/pipeline/dropped_examples"] = self._dropped_examples
        values["data/pipeline/exhausted"] = int(self._exhausted)
        return values

    def load_state_dict(self, state):
        expected = {
            "source",
            "packer",
            "transforms",
            "model_io",
            "supervision",
            "collator",
            "ready",
            "exhausted",
            "dropped_examples",
            "worker_topology",
        }
        if set(state) != expected:
            raise CheckpointError(
                "data pipeline checkpoint state is incomplete or has unknown fields"
            )
        ready = state["ready"]
        if not isinstance(ready, (tuple, list)) or any(
            not isinstance(item, SupervisedExample) for item in ready
        ):
            raise CheckpointError(
                "data pipeline ready buffer must contain supervised examples"
            )
        exhausted = state["exhausted"]
        dropped_examples = state["dropped_examples"]
        if (
            not isinstance(exhausted, bool)
            or not isinstance(dropped_examples, int)
            or isinstance(dropped_examples, bool)
            or dropped_examples < 0
        ):
            raise CheckpointError("data pipeline finite-source state is invalid")
        transform_states = state["transforms"]
        if not isinstance(transform_states, (tuple, list)) or len(
            transform_states
        ) != len(self.transforms):
            raise CheckpointError("data transform state count mismatch")
        saved_topology = state["worker_topology"]
        if saved_topology is not None:
            if (
                not isinstance(saved_topology, (tuple, list))
                or len(saved_topology) != 2
            ):
                raise CheckpointError("data pipeline worker topology is invalid")
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in saved_topology
            ):
                raise CheckpointError("data pipeline worker topology is invalid")
            saved_topology = tuple(saved_topology)
            if saved_topology[1] <= 0 or not 0 <= saved_topology[0] < saved_topology[1]:
                raise CheckpointError("data pipeline worker topology is invalid")
        if self._loader_batch_size is not None and saved_topology is not None:
            worker = get_worker_info()
            worker_id = 0 if worker is None else worker.id
            num_workers = 1 if worker is None else worker.num_workers
            if saved_topology != (worker_id, num_workers):
                raise CheckpointError("data pipeline worker topology changed")
            self.shard(
                rank=self._loader_rank,
                world_size=self._loader_world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            self._worker_topology = (worker_id, num_workers)
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
        self._exhausted = exhausted
        self._dropped_examples = dropped_examples
        self._worker_topology = saved_topology


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
        drop_last=spec.drop_last,
    )
