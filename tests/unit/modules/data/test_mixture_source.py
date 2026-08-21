from copy import deepcopy

import pytest

from trainomni.core.errors import CheckpointError, SpecError
from trainomni.modules.data.sources.memory.config import MemorySourceConfig
from trainomni.modules.data.sources.memory.module import MemorySource
from trainomni.modules.data.sources.mixture.config import MixtureSourceConfig
from trainomni.modules.data.sources.mixture.module import MixtureSource


def memory_source(name: str, *, repeat: bool = True) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(
            samples=(
                {
                    "sample_id": name,
                    "content": ({"kind": "text", "value": name},),
                },
            ),
            repeat=repeat,
        )
    )


def build_mixture(*, seed: int = 7, weights=None) -> MixtureSource:
    return MixtureSource(
        MixtureSourceConfig(
            weights=weights or {"alpha": 1.0, "beta": 3.0},
            seed=seed,
        ),
        {"alpha": memory_source("a"), "beta": memory_source("b")},
    )


def test_weighted_selection_and_all_child_cursors_resume_exactly() -> None:
    source = build_mixture()
    prefix = [source.next_sample() for _ in range(9)]
    state = source.state_dict()
    expected = [source.next_sample().sample_id for _ in range(20)]

    restored = build_mixture()
    restored.load_state_dict(state)
    actual = [restored.next_sample().sample_id for _ in range(20)]
    assert actual == expected
    assert all("::" in sample.sample_id for sample in prefix)
    assert all(sample.metadata["trainomni.source"] in {"alpha", "beta"} for sample in prefix)
    assert sum(restored.counts.values()) == restored.cursor == 29
    assert restored.counts["beta"] > restored.counts["alpha"]
    metrics = restored.metrics()
    assert metrics["data/mixture/samples"] == 29
    assert metrics["data/source/alpha/samples"] == restored.counts["alpha"]


def test_mixture_identity_and_state_corruption_fail_closed() -> None:
    source = build_mixture()
    source.next_sample()
    state = source.state_dict()

    with pytest.raises(CheckpointError, match="identity changed"):
        build_mixture(seed=8).load_state_dict(state)

    corrupted = deepcopy(state)
    corrupted["counts"]["alpha"] += 1
    with pytest.raises(CheckpointError, match="do not sum"):
        build_mixture().load_state_dict(corrupted)


def test_mixture_requires_exact_named_source_set() -> None:
    with pytest.raises(SpecError, match="names differ"):
        MixtureSource(
            MixtureSourceConfig(weights={"alpha": 1.0}),
            {"beta": memory_source("b")},
        )


def test_finite_children_are_removed_deterministically_and_resume() -> None:
    config = MixtureSourceConfig(weights={"alpha": 1.0, "beta": 1.0}, seed=3)
    source = MixtureSource(
        config,
        {
            "alpha": memory_source("a", repeat=False),
            "beta": memory_source("b", repeat=False),
        },
    )
    first = source.next_sample().sample_id
    state = source.state_dict()
    second = source.next_sample().sample_id
    with pytest.raises(StopIteration):
        source.next_sample()

    restored = MixtureSource(
        config,
        {
            "alpha": memory_source("a", repeat=False),
            "beta": memory_source("b", repeat=False),
        },
    )
    restored.load_state_dict(state)
    assert restored.next_sample().sample_id == second
    with pytest.raises(StopIteration):
        restored.next_sample()
    assert {first, second} == {"alpha::a", "beta::b"}
