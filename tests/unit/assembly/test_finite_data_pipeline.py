from __future__ import annotations

from dataclasses import dataclass

import pytest

from trainomni.assembly.data_builder import DataPipelineStream
from trainomni.core.errors import CheckpointError, SpecError


@dataclass
class FiniteSource:
    values: tuple[int, ...]
    cursor: int = 0
    is_finite: bool = True

    def next_sample(self):
        if self.cursor >= len(self.values):
            raise StopIteration
        value = self.values[self.cursor]
        self.cursor += 1
        return value

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        if set(state) != {"cursor"}:
            raise CheckpointError("source state is invalid")
        self.cursor = int(state["cursor"])


class StatefulIdentity:
    def apply(self, value):
        return value

    def encode(self, value):
        return value

    def annotate(self, value):
        return value

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise CheckpointError("identity state is invalid")


class ImmediatePacker:
    def add(self, value):
        return (value,)

    def flush(self):
        return ()

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise CheckpointError("packer state is invalid")


class BufferedPacker(ImmediatePacker):
    def __init__(self):
        self.buffer = []

    def add(self, value):
        self.buffer.append(value)
        return ()

    def flush(self):
        values = tuple(self.buffer)
        self.buffer.clear()
        return values

    def state_dict(self):
        return {"buffer": tuple(self.buffer)}

    def load_state_dict(self, state):
        if set(state) != {"buffer"}:
            raise CheckpointError("packer state is invalid")
        self.buffer = list(state["buffer"])


class Collator:
    def collate(self, values):
        return tuple(values)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise CheckpointError("collator state is invalid")


def stream(*, drop_last: bool, packer=None):
    identity = StatefulIdentity()
    return DataPipelineStream(
        source=FiniteSource((0, 1, 2)),
        transforms=(identity,),
        model_io=identity,
        supervision=identity,
        packer=packer or ImmediatePacker(),
        collator=Collator(),
        drop_last=drop_last,
    )


def test_finite_stream_returns_or_explicitly_drops_a_partial_batch() -> None:
    keep = stream(drop_last=False)
    assert keep.next_batch(2) == (0, 1)
    state = keep.state_dict()
    assert keep.next_batch(2) == (2,)
    with pytest.raises(StopIteration):
        keep.next_batch(2)

    resumed = stream(drop_last=False)
    resumed.load_state_dict(state)
    assert resumed.next_batch(2) == (2,)
    with pytest.raises(StopIteration):
        resumed.next_batch(2)

    drop = stream(drop_last=True)
    assert drop.next_batch(2) == (0, 1)
    with pytest.raises(StopIteration):
        drop.next_batch(2)
    assert drop.metrics()["data/pipeline/dropped_examples"] == 1
    assert drop.metrics()["data/pipeline/exhausted"] == 1


def test_finite_stream_flushes_a_buffering_packer_before_eof() -> None:
    buffered = stream(drop_last=False, packer=BufferedPacker())
    assert buffered.next_batch(4) == (0, 1, 2)
    with pytest.raises(StopIteration):
        buffered.next_batch(1)


@pytest.mark.parametrize("finite", [True, None])
def test_multi_rank_unknown_or_finite_exhaustion_fails_closed(finite) -> None:
    candidate = stream(drop_last=False)
    candidate.source.is_finite = finite
    with pytest.raises(SpecError, match="equal optimizer steps"):
        candidate.shard(rank=0, world_size=2)
