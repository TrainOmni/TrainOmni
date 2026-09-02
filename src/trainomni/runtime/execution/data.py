"""Deterministic rank sharding below transforms and collation."""

from __future__ import annotations

from collections.abc import Mapping

from trainomni.core.errors import CheckpointError


class RankShardedSource:
    def __init__(self, source, *, rank: int, world_size: int) -> None:
        if world_size <= 1 or not 0 <= rank < world_size:
            raise ValueError("rank sharding requires world_size > 1 and an in-range rank")
        self.source = source
        self.rank = rank
        self.world_size = world_size
        self.local_samples = 0

    def next_sample(self):
        selected = None
        for index in range(self.world_size):
            sample = self.source.next_sample()
            if index == self.rank:
                selected = sample
        if selected is None:  # pragma: no cover - constructor prevents this
            raise RuntimeError("rank sharding selected no sample")
        self.local_samples += 1
        return selected

    def metrics(self):
        hook = getattr(self.source, "metrics", None)
        values = {} if not callable(hook) else dict(hook())
        values.update(
            {
                "distributed_rank": self.rank,
                "distributed_world_size": self.world_size,
                "distributed_local_samples": self.local_samples,
            }
        )
        return values

    def state_dict(self):
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_samples": self.local_samples,
            "source": dict(self.source.state_dict()),
        }

    def load_state_dict(self, state: Mapping) -> None:
        expected = {"rank", "world_size", "local_samples", "source"}
        if set(state) != expected:
            raise CheckpointError("distributed source state keys are invalid")
        if any(
            not isinstance(state[field], int) or isinstance(state[field], bool)
            for field in ("rank", "world_size", "local_samples")
        ):
            raise CheckpointError("distributed source counters must be integers")
        if state["rank"] != self.rank or state["world_size"] != self.world_size:
            raise CheckpointError("distributed source topology changed")
        local_samples = state["local_samples"]
        if local_samples < 0:
            raise CheckpointError("distributed source local sample count is invalid")
        self.source.load_state_dict(state["source"])
        self.local_samples = local_samples
