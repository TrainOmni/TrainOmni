"""Deterministic rank sharding at the already-planned batch boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainomni.models import ModelBatch

DISTRIBUTED_BATCH_STATE_VERSION = "trainomni.distributed-batch.v1"


class DistributedBatchStream:
    """Consume the same global batch sequence on every rank, return one shard.

    Grouping after cost-aware planning avoids rank drift caused by independently
    packing variable-size multimodal samples. All ranks advance the underlying
    stream by exactly ``world_size`` batches per optimizer microstep.
    """

    def __init__(self, batches: Any, *, rank: int, world_size: int) -> None:
        if world_size <= 1:
            raise ValueError("DistributedBatchStream requires world_size > 1")
        if not 0 <= rank < world_size:
            raise ValueError("distributed rank is outside world size")
        if not callable(getattr(batches, "state_dict", None)) or not callable(
            getattr(batches, "load_state_dict", None)
        ):
            raise TypeError("distributed child batch stream must be stateful")
        self.batches = batches
        self.rank = rank
        self.world_size = world_size
        self._groups = 0

    def __iter__(self) -> DistributedBatchStream:
        return self

    def __next__(self) -> ModelBatch:
        group = []
        iterator = iter(self.batches)
        for _ in range(self.world_size):
            try:
                group.append(next(iterator))
            except StopIteration:
                if not group:
                    raise
                # Dropping an incomplete global group keeps collective step counts
                # identical across ranks.
                raise StopIteration
        self._groups += 1
        return group[self.rank]

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": DISTRIBUTED_BATCH_STATE_VERSION,
            "rank": self.rank,
            "world_size": self.world_size,
            "groups": self._groups,
            "batches": self.batches.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "state_version": DISTRIBUTED_BATCH_STATE_VERSION,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"distributed batch {key} mismatch")
        groups = state.get("groups")
        child = state.get("batches")
        if not isinstance(groups, int) or groups < 0 or not isinstance(child, Mapping):
            raise ValueError("distributed batch state is invalid")
        self.batches.load_state_dict(child)
        self._groups = groups
