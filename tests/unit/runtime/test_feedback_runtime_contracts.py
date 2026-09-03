import json
import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import IterableDataset

from trainomni.artifacts.manifest import _write_exact
from trainomni.core.errors import CheckpointError, DataLoadingError, SpecError
from trainomni.core.module import ModuleRef, parse_config
from trainomni.modules.model.attention.packed.config import PackedAttentionConfig
from trainomni.modules.model.attention.packed.module import PackedAttentionPolicy
from trainomni.modules.model.language.transformers_causal_lm.module import (
    TransformersCausalLanguage,
)
from trainomni.runtime.data_loader import StatefulBatchLoader
from trainomni.runtime.observability.events import JsonlEventWriter
from trainomni.specs.run import DataLoaderSpec


class ExitWorker(IterableDataset):
    def configure_loader(self, **kwargs):
        pass

    def __iter__(self):
        os._exit(17)  # Only this test's explicitly spawned worker exits.
        yield  # pragma: no cover


@dataclass(frozen=True)
class NestedAssetConfig:
    assets: Mapping[str, Mapping[str, str]]


def test_resolved_nested_extension_config_is_immutable_and_pickle_safe():
    original = {"assets": {"vision": {"config.json": "a" * 64}}}
    ref = ModuleRef.from_mapping({
        "module": "model_io:example/assets@1", "config": original,
    }, field_name="model_io")
    original["assets"]["vision"]["config.json"] = "b" * 64
    config = parse_config(NestedAssetConfig, ref.config)
    restored = pickle.loads(pickle.dumps(config))
    assert restored.assets["vision"]["config.json"] == "a" * 64
    with pytest.raises(TypeError):
        restored.assets["vision"]["config.json"] = "changed"
    assert pickle.loads(pickle.dumps(ref)) == ref


def test_spawn_worker_death_is_an_actionable_error_not_a_fallback():
    loader = StatefulBatchLoader(
        ExitWorker(), batch_size=1, rank=0, world_size=1,
        spec=DataLoaderSpec(num_workers=1, multiprocessing_context="spawn", timeout_seconds=15),
    )
    try:
        with pytest.raises(DataLoadingError, match="rank 0.*workers=1.*start_method=spawn"):
            loader.next_batch(1)
        assert loader.metrics()["data/loader/batches"] == 0
    finally:
        loader.close()


@pytest.mark.parametrize("values", [
    {"multiprocessing_context": "unknown"}, {"timeout_seconds": -1},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": True},
    {"num_workers": 0, "timeout_seconds": 1},
])
def test_invalid_worker_controls_fail_at_config_parse(values):
    with pytest.raises(SpecError):
        DataLoaderSpec.from_mapping(values)


def test_default_worker_context_does_not_inherit_fork():
    assert DataLoaderSpec.from_mapping({"num_workers": 2}).multiprocessing_context == "spawn"


def test_wait_metric_keeps_backward_compatible_cumulative_semantics():
    loader = object.__new__(StatefulBatchLoader)
    loader.batch_size = 1
    loader._exhausted = False
    loader._wait_seconds = 0.0
    loader._batches = loader._samples = 0
    loader.spec = DataLoaderSpec()
    loader.pipeline = SimpleNamespace()
    loader._iterator = iter([SimpleNamespace(sample_ids=("a",)), SimpleNamespace(sample_ids=("b",))])
    with patch("trainomni.runtime.data_loader.time.perf_counter", side_effect=[1.0, 1.1, 2.0, 2.2]):
        loader.next_batch(1)
        first = loader.metrics()["data/loader/wait_seconds_total"]
        loader.next_batch(1)
    assert first == pytest.approx(0.1)
    assert loader.metrics()["data/loader/wait_seconds"] == pytest.approx(0.3)
    assert loader.metrics()["data/loader/wait_seconds_total"] == pytest.approx(0.3)


def test_event_rows_include_utc_and_monotonic_timing(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JsonlEventWriter(path)
    sink.write("optimizer_step", {"global_step": 1})
    sink.write("optimizer_step", {"global_step": 2})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(datetime.fromisoformat(row["timestamp"]).utcoffset().total_seconds() == 0 for row in rows)
    assert rows[1]["monotonic_seconds"] >= rows[0]["monotonic_seconds"]


def test_attention_error_names_actual_shape_and_head_axis():
    policy = PackedAttentionPolicy(PackedAttentionConfig())
    with pytest.raises(SpecError, match=r"got shape=\(2, 1, 1, 4, 4\).*head axis"):
        policy.apply(
            input_ids=torch.ones(2, 4, dtype=torch.long), attention_mask=None,
            modal_positions=None,
            model_inputs={"packed_attention_mask": torch.ones(2, 1, 1, 4, 4, dtype=torch.bool)},
        )


def test_new_run_does_not_require_deleting_old_outputs(tmp_path):
    old = tmp_path / "old" / "resolved" / "task.resolved.json"
    _write_exact(old, {"task": "old"})
    original = old.read_bytes()
    with pytest.raises(CheckpointError, match="new checkpoint.directory.*keep the old outputs"):
        _write_exact(old, {"task": "new"})
    new = tmp_path / "new" / "resolved" / "task.resolved.json"
    _write_exact(new, {"task": "new"})
    assert old.read_bytes() == original
    assert json.loads(new.read_text()) == {"task": "new"}


@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bool])
def test_additive_attention_bias_matches_embedding_precision(mask_dtype):
    class Capture(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs
    language = TransformersCausalLanguage(Capture())
    mask = torch.ones(1, 1, 3, 3, dtype=mask_dtype)
    if mask_dtype == torch.float32:
        mask[:, :, -1] = torch.finfo(torch.float32).min
    result = language.forward_embeddings(torch.ones(1, 3, 8, dtype=torch.bfloat16), attention_mask=mask)
    expected = torch.bfloat16 if mask_dtype == torch.float32 else torch.bool
    assert result["attention_mask"].dtype == expected
    assert mask.dtype == mask_dtype
    assert torch.isfinite(result["attention_mask"]).all()
