"""Finite streams never acquire a new epoch through checkpoint restoration."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

pa = pytest.importorskip("pyarrow")
from pyarrow import parquet

from trainomni.assembly.data_builder import DataPipelineStream
from trainomni.contracts.batch import EncodedSample
from trainomni.core.context import BuildContext
from trainomni.core.errors import CheckpointError
from trainomni.core.module import ModuleRef
from trainomni.modules.data.adapters.binding import AdaptedSource
from trainomni.modules.data.adapters.msswift.config import MSSwiftAdapterConfig
from trainomni.modules.data.adapters.msswift.module import MSSwiftAdapter
from trainomni.modules.data.collation.multimodal.config import (
    MultimodalCollatorConfig,
)
from trainomni.modules.data.collation.multimodal.module import MultimodalCollator
from trainomni.modules.data.packing.none.module import NoPacker
from trainomni.modules.data.sources.parquet.module import descriptor
from trainomni.modules.data.supervision.causal_lm.config import (
    CausalSupervisionConfig,
)
from trainomni.modules.data.supervision.causal_lm.module import CausalSupervision
from trainomni.runtime.data_loader import build_stateful_batch_loader
from trainomni.specs.run import DataLoaderSpec


class TextModelIO:
    def encode(self, sample):
        return EncodedSample(sample.sample_id, {"input_ids": torch.tensor([1, 2, 3])})


def make_loader(path, batch_size, workers, persistent=False):
    reference = ModuleRef.from_mapping(
        {
            "module": "data_source:trainomni/parquet@1",
            "config": {
                "dataset_id": "finite-tail", "paths": [str(path)], "repeat": False,
            },
        },
        field_name="data.source",
    )
    source = descriptor().build(reference, BuildContext("finite-tail"))
    return build_stateful_batch_loader(
        DataPipelineStream(
            source=AdaptedSource(source, MSSwiftAdapter(MSSwiftAdapterConfig())),
            transforms=(),
            model_io=TextModelIO(),
            supervision=CausalSupervision(CausalSupervisionConfig()),
            packer=NoPacker(),
            collator=MultimodalCollator(MultimodalCollatorConfig()),
            drop_last=False,
        ),
        batch_size=batch_size,
        spec=DataLoaderSpec(
            num_workers=workers,
            prefetch_factor=2 if workers else None,
            persistent_workers=persistent,
        ),
    )


def write_rows(path, count, workers):
    parquet.write_table(
        pa.Table.from_pylist(
            [{"id": f"row-{index}", "text": "test"} for index in range(count)]
        ),
        path,
        row_group_size=count // max(workers, 1),
    )


@pytest.mark.parametrize(
    "count,batch_size,workers,persistent",
    [(2, 2, 2, False), (6, 4, 2, False), (3, 2, 0, False),
     (4, 2, 0, False), (4, 2, 2, False), (6, 4, 2, True), (4, 2, 2, True)],
)
@pytest.mark.parametrize("observe_eof", [False, True])
def test_restore_last_batch_or_observed_eof_stays_finished(
    tmp_path, count, batch_size, workers, persistent, observe_eof,
):
    path = tmp_path / "rows.parquet"
    write_rows(path, count, workers)
    first = make_loader(path, batch_size, workers, persistent)
    restored = None
    try:
        partitions = max(workers, 1)
        batch_count = partitions * ((count // partitions + batch_size - 1) // batch_size)
        batches = [first.next_batch(batch_size) for _ in range(batch_count)]
        assert sum(len(batch.sample_ids) for batch in batches) == count
        if observe_eof:
            with pytest.raises(StopIteration):
                first.next_batch(batch_size)
        state = deepcopy(first.state_dict())
        assert state["exhausted"] is observe_eof
        first.close()

        restored = make_loader(path, batch_size, workers, persistent)
        restored.load_state_dict(state)
        # Taking another snapshot immediately must neither consume data nor
        # construct an upstream iterator that could reset a finished epoch.
        resaved = restored.state_dict()
        assert restored._iterator is None
        restored.load_state_dict(resaved)
        for _ in range(2):
            with pytest.raises(StopIteration):
                restored.next_batch(batch_size)
        assert restored.state_dict()["exhausted"] is True
        assert restored.metrics()["data/loader/samples"] == count
        assert restored.metrics()["data/loader/batches"] == batch_count
    finally:
        first.close()
        if restored is not None:
            restored.close()


def test_v1_terminal_state_migrates_without_starting_an_epoch(tmp_path):
    path = tmp_path / "legacy.parquet"
    write_rows(path, 1, 0)
    first = make_loader(path, 2, 0)
    restored = make_loader(path, 2, 0)
    try:
        first.next_batch(2)
        with pytest.raises(StopIteration):
            first.next_batch(2)
        state = first.state_dict()
        state["schema_version"] = 1
        del state["exhausted"]
        restored.load_state_dict(state)
        with pytest.raises(StopIteration):
            restored.next_batch(2)
        assert restored.state_dict()["schema_version"] == 2
        assert restored._iterator is None
        del state["loader"]["_iterator_finished"]
        with pytest.raises(CheckpointError, match="exhausted"):
            restored.load_state_dict(state)
    finally:
        first.close()
        restored.close()


@pytest.mark.parametrize(
    "changes",
    [{"batches": True}, {"samples": 1.5}, {"batches": -1},
     {"batches": 0, "samples": 1}, {"batches": 2, "samples": 1},
     {"batches": 1, "samples": 3}, {"wait_seconds": float("nan")},
     {"exhausted": 1}, {"exhausted": True}, {"schema_version": 2.0}],
)
def test_invalid_counters_and_terminal_markers_fail_closed(tmp_path, changes):
    path = tmp_path / "counters.parquet"
    write_rows(path, 1, 0)
    loader = make_loader(path, 2, 0)
    try:
        state = loader.state_dict()
        state.update(changes)
        with pytest.raises(CheckpointError):
            loader.load_state_dict(state)
    finally:
        loader.close()


def test_finalized_finite_training_and_evaluation_do_not_replay(tmp_path):
    from trainomni.modules.objectives.causal_lm.config import CausalLMConfig
    from trainomni.modules.objectives.causal_lm.module import CausalLMObjective
    from trainomni.modules.parameters.full.config import FullParameterConfig
    from trainomni.modules.parameters.full.module import FullParameterPolicy
    from trainomni.runtime.evaluation import evaluate_batches
    from trainomni.runtime.loop.engine import TrainEngine
    from trainomni.specs.run import RunSpec

    class CountingLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8)
            self.calls = 0

        def forward(self, input_ids, **kwargs):
            self.calls += 1
            return SimpleNamespace(logits=self.head(self.embed(input_ids)))

    path = tmp_path / "finite.parquet"
    write_rows(path, 3, 0)
    run = RunSpec.from_mapping({
        "schema_version": 1, "name": "finite-finalized", "max_steps": 2,
        "device": "cpu", "precision": "fp32", "per_device_batch_size": 2,
        "checkpoint": {"directory": str(tmp_path / "checkpoints"), "every_steps": 2},
    })

    def build_engine():
        model = CountingLM()
        return TrainEngine(
            model=model,
            objective=CausalLMObjective(CausalLMConfig()),
            parameter_selection=FullParameterPolicy(FullParameterConfig()).apply(model),
            stream=make_loader(path, 2, 0).pipeline,
            run=run, task_digest="finite-finalized", module_lock={},
        )

    first = build_engine()
    try:
        assert len(first.train()) == 2
        assert first.stream.metrics()["data/loader/samples"] == 3
    finally:
        first.close()
    restored = build_engine()
    try:
        restored.resume(tmp_path / "checkpoints" / "step-00000002")
        assert restored.train() == ()
        assert restored.model.calls == 0
        with pytest.raises(StopIteration):
            restored.stream.next_batch(2)
        # Evaluation must propagate finite exhaustion rather than forwarding
        # replayed data. A fresh evaluation API call still owns a fresh stream.
        with pytest.raises(StopIteration):
            evaluate_batches(
                model=restored.model, objective=restored.objective,
                stream=restored.stream,
                evaluators=(SimpleNamespace(reset=lambda: None),),
                device=restored.device, batches=1, batch_size=2,
            )
        assert restored.model.calls == 0
    finally:
        restored.close()
