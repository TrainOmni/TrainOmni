"""Bind a physical record source to one semantic data adapter."""

from __future__ import annotations

from collections.abc import Mapping

from trainomni.core.errors import CheckpointError, SpecError


def _optional_state(owner):
    hook = getattr(owner, "state_dict", None)
    return None if not callable(hook) else dict(hook())


class AdaptedSource:
    def __init__(self, source, adapter) -> None:
        if not callable(getattr(source, "next_record", None)):
            raise SpecError("adapted data source must implement next_record")
        if not callable(getattr(adapter, "adapt", None)):
            raise SpecError("data adapter must implement adapt")
        self.source = source
        self.adapter = adapter

    def next_sample(self):
        return self.adapter.adapt(self.source.next_record())

    def shard(
        self,
        *,
        rank: int,
        world_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> None:
        hook = getattr(self.source, "shard", None)
        if not callable(hook):
            raise SpecError(
                "record source does not support physical rank sharding; "
                "refusing post-read sample discard"
            )
        hook(
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )

    def state_dict(self):
        return {
            "source": dict(self.source.state_dict()),
            "adapter": _optional_state(self.adapter),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if set(state) != {"source", "adapter"}:
            raise CheckpointError("invalid adapted-source state keys")
        self.source.load_state_dict(state["source"])
        adapter_state = state["adapter"]
        hook = getattr(self.adapter, "load_state_dict", None)
        if adapter_state is None:
            if callable(hook):
                hook({})
        elif not callable(hook):
            raise CheckpointError("adapter checkpoint state has no owner")
        else:
            hook(adapter_state)

    def metrics(self):
        hook = getattr(self.source, "metrics", None)
        return {} if not callable(hook) else dict(hook())
