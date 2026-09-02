"""Default stateful PyTorch batch-loader runtime."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from numbers import Real
from typing import Any

from trainomni.core.errors import CheckpointError, SpecError
from trainomni.specs.run import DataLoaderSpec


class StatefulBatchLoader:
    """Expose StatefulDataLoader through the framework batch-stream contract."""

    def __init__(
        self,
        pipeline,
        *,
        batch_size: int,
        spec: DataLoaderSpec,
        rank: int,
        world_size: int,
    ) -> None:
        try:
            from torchdata.stateful_dataloader import StatefulDataLoader
        except ImportError as exc:  # pragma: no cover - packaging contract
            raise SpecError(
                "the default data runtime requires torchdata>=0.11,<0.12"
            ) from exc
        pipeline.configure_loader(
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": None,
            "num_workers": spec.num_workers,
            "persistent_workers": spec.persistent_workers,
            "pin_memory": spec.pin_memory,
            "in_order": spec.in_order,
            "snapshot_every_n_steps": spec.snapshot_every_n_steps,
        }
        if spec.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = spec.prefetch_factor
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.spec = spec
        self.loader = StatefulDataLoader(pipeline, **loader_kwargs)
        self._iterator = None
        self._batches = 0
        self._samples = 0
        self._wait_seconds = 0.0
        self._exhausted = False
        self._pending_loader_state = None

    def _ensure_iterator(self):
        if self._iterator is None:
            self._iterator = iter(self.loader)
            self._pending_loader_state = None
        return self._iterator

    def next_batch(self, batch_size: int):
        if batch_size != self.batch_size:
            raise SpecError(
                "data loader batch size is immutable for one run: "
                f"configured {self.batch_size}, requested {batch_size}"
            )
        if self._exhausted:
            raise StopIteration
        started = time.perf_counter()
        try:
            batch = next(self._ensure_iterator())
        except StopIteration:
            # TrainOmni streams stay exhausted. Upstream __iter__ deliberately
            # starts a new epoch when restoring a finished iterator.
            self._exhausted = True
            raise
        self._wait_seconds += time.perf_counter() - started
        self._batches += 1
        sample_ids = getattr(batch, "sample_ids", None)
        self._samples += self.batch_size if sample_ids is None else len(sample_ids)
        return batch

    def state_dict(self):
        loader_state = self._pending_loader_state
        if loader_state is None:
            loader_state = self.loader.state_dict()
        return {
            "schema_version": 2,
            "batch_size": self.batch_size,
            "loader": dict(loader_state),
            "batches": self._batches,
            "samples": self._samples,
            "wait_seconds": self._wait_seconds,
            "exhausted": self._exhausted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "batch_size",
            "loader",
            "batches",
            "samples",
            "wait_seconds",
        }
        version = state.get("schema_version")
        if type(version) is not int or version not in {1, 2}:
            raise CheckpointError("invalid stateful data loader state")
        if version == 2:
            expected.add("exhausted")
        if set(state) != expected:
            raise CheckpointError("invalid stateful data loader state")
        if not isinstance(state["batch_size"], int) or isinstance(
            state["batch_size"], bool
        ):
            raise CheckpointError("data loader batch size state must be an integer")
        if state["batch_size"] != self.batch_size:
            raise CheckpointError("data loader batch size changed")
        batches = state["batches"]
        samples = state["samples"]
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (batches, samples)
        ):
            raise CheckpointError("data loader counters must be integers")
        wait_seconds = state["wait_seconds"]
        if not isinstance(wait_seconds, Real) or isinstance(wait_seconds, bool):
            raise CheckpointError("data loader wait_seconds must be numeric")
        wait_seconds = float(wait_seconds)
        if min(batches, samples, wait_seconds) < 0 or not math.isfinite(wait_seconds):
            raise CheckpointError("data loader counters must be non-negative")
        if (batches == 0) != (samples == 0):
            raise CheckpointError("data loader batch/sample counters are inconsistent")
        # Each worker may emit its own partial tail; there is not necessarily
        # just one partial batch across the whole loader.
        if not batches <= samples <= batches * self.batch_size:
            raise CheckpointError("data loader batch/sample counters are inconsistent")
        if not isinstance(state["loader"], Mapping):
            raise CheckpointError("data loader implementation state must be a mapping")
        # v1 did not own terminal state. Its pinned TorchData 0.11 payload has
        # this marker in both single-process and multiprocessing snapshots.
        upstream_finished = state["loader"].get("_iterator_finished")
        exhausted = upstream_finished if version == 1 else state["exhausted"]
        if not isinstance(exhausted, bool):
            raise CheckpointError("data loader exhausted state must be a boolean")
        if upstream_finished is not None and (
            not isinstance(upstream_finished, bool) or upstream_finished != exhausted
        ):
            raise CheckpointError("data loader exhausted state disagrees with its iterator")
        self.close()
        self._pending_loader_state = dict(state["loader"])
        self.loader.load_state_dict(self._pending_loader_state)
        self._iterator = None
        self._batches = batches
        self._samples = samples
        self._wait_seconds = wait_seconds
        self._exhausted = exhausted

    def metrics(self):
        values = {
            "data/loader/batches": self._batches,
            "data/loader/samples": self._samples,
            "data/loader/wait_seconds": self._wait_seconds,
            "data/loader/num_workers": self.spec.num_workers,
            "data/loader/prefetch_factor": self.spec.prefetch_factor or 0,
            "data/loader/pin_memory": int(self.spec.pin_memory),
            "data/loader/exhausted": int(self._exhausted),
        }
        if self.spec.num_workers == 0:
            hook = getattr(self.pipeline, "metrics", None)
            if callable(hook):
                values.update(hook())
        return values

    def close(self) -> None:
        iterator = self._iterator
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        self._iterator = None


def build_stateful_batch_loader(
    pipeline,
    *,
    batch_size: int,
    spec: DataLoaderSpec,
    rank: int = 0,
    world_size: int = 1,
) -> StatefulBatchLoader:
    if not callable(getattr(pipeline, "configure_loader", None)):
        if spec.num_workers != 0:
            raise SpecError(
                "multi-worker loading requires a DataPipelineStream; custom batch "
                "streams can only use data_loader.num_workers=0"
            )
        if world_size > 1:
            shard = getattr(pipeline, "shard", None)
            if not callable(shard):
                raise SpecError(
                    "multi-rank execution requires a rank-shardable batch stream"
                )
            shard(rank=rank, world_size=world_size)
        return pipeline
    return StatefulBatchLoader(
        pipeline,
        batch_size=batch_size,
        spec=spec,
        rank=rank,
        world_size=world_size,
    )
