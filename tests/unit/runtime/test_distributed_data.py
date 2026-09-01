from __future__ import annotations

from dataclasses import dataclass

import pytest

from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.errors import CheckpointError
from trainomni.runtime.execution.data import RankShardedSource
from trainomni.runtime.execution.process import ProcessContext


@dataclass
class Source:
    cursor: int = 0

    def next_sample(self):
        sample = OmniSample(
            sample_id=f"sample-{self.cursor}",
            content=(ContentBlock("text", str(self.cursor)),),
        )
        self.cursor += 1
        return sample

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])


def test_rank_sharding_is_disjoint_deterministic_and_resumable() -> None:
    rank0 = RankShardedSource(Source(), rank=0, world_size=2)
    rank1 = RankShardedSource(Source(), rank=1, world_size=2)
    assert [rank0.next_sample().sample_id for _ in range(3)] == [
        "sample-0",
        "sample-2",
        "sample-4",
    ]
    assert [rank1.next_sample().sample_id for _ in range(3)] == [
        "sample-1",
        "sample-3",
        "sample-5",
    ]
    state = rank1.state_dict()
    restored = RankShardedSource(Source(), rank=1, world_size=2)
    restored.load_state_dict(state)
    assert restored.next_sample().sample_id == "sample-7"
    with pytest.raises(CheckpointError, match="topology changed"):
        RankShardedSource(Source(), rank=0, world_size=2).load_state_dict(state)


def test_world_size_one_metrics_preserve_rank_structure() -> None:
    process = ProcessContext("single", 0, 0, 1, None, False)

    assert process.all_gather_metrics({"samples": 3, "ratio": 0.5}) == (
        {"samples": 3, "ratio": 0.5},
    )
