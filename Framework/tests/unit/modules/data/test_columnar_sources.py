from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

pa = pytest.importorskip("pyarrow")
from pyarrow import ipc, parquet

from trainomni.assembly.data_builder import build_data_stream
from trainomni.contracts.batch import EncodedSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.context import BuildContext
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId, ModuleRef
from trainomni.core.registry import ModuleRegistry
from trainomni.core.resolver import ModuleResolver
from trainomni.modules.data.adapters.binding import AdaptedSource
from trainomni.modules.data.adapters.msswift.module import (
    descriptor as adapter_descriptor,
)
from trainomni.modules.data.collation.multimodal.module import (
    descriptor as collator_descriptor,
)
from trainomni.modules.data.packing.none.module import (
    descriptor as packer_descriptor,
)
from trainomni.modules.data.sources.arrow.module import (
    descriptor as arrow_descriptor,
)
from trainomni.modules.data.sources.parquet.module import (
    descriptor as parquet_descriptor,
)
from trainomni.modules.data.supervision.causal_lm.module import (
    descriptor as supervision_descriptor,
)
from trainomni.specs.task import DataPipelineSpec


def image_bytes() -> bytes:
    from PIL import Image

    payload = io.BytesIO()
    Image.new("RGB", (4, 3), color=(12, 34, 56)).save(payload, format="PNG")
    return payload.getvalue()


def rows(count: int) -> list[dict]:
    payload = image_bytes()
    return [
        {
            "id": f"row-{index}",
            "messages": [
                {"role": "user", "content": f"<image>question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
            "images": [{"bytes": payload, "path": None}],
            "quality": index / 10,
        }
        for index in range(count)
    ]


def build_adapter():
    reference = ModuleRef.from_mapping(
        {
            "module": "data_adapter:trainomni/msswift@1",
            "config": {"metadata_columns": ["quality"]},
        },
        field_name="data.adapter",
    )
    return adapter_descriptor().build(reference, BuildContext("task"))


def build_parquet(path: Path, *, repeat: bool = False):
    reference = ModuleRef.from_mapping(
        {
            "module": "data_source:trainomni/parquet@1",
            "config": {
                "dataset_id": "parquet-fixture",
                "paths": [str(path)],
                "batch_rows": 2,
                "repeat": repeat,
            },
        },
        field_name="data.source",
    )
    return parquet_descriptor().build(reference, BuildContext("task"))


def build_arrow(path: Path, *, repeat: bool = False):
    reference = ModuleRef.from_mapping(
        {
            "module": "data_source:trainomni/arrow@1",
            "config": {
                "dataset_id": "arrow-fixture",
                "paths": [str(path)],
                "batch_rows": 2,
                "repeat": repeat,
            },
        },
        field_name="data.source",
    )
    return arrow_descriptor().build(reference, BuildContext("task"))


def collect(source) -> list:
    values = []
    while True:
        try:
            values.append(source.next_sample())
        except StopIteration:
            return values


@dataclass(frozen=True)
class FixtureModelIOConfig:
    pass


class FixtureModelIO:
    def encode(self, sample):
        image = next(
            block.value
            for message in sample.messages
            for block in message.content
            if block.kind == "image"
        )
        return EncodedSample(
            sample.sample_id,
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "pixel_values": torch.tensor(
                    list(image.tobytes()), dtype=torch.float32
                ).reshape(-1, 3),
            },
        )


def model_io_descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("model_io:test/columnar@1"),
        config_type=FixtureModelIOConfig,
        factory=lambda config, context: FixtureModelIO(),
        provides=CapabilitySet.of({"data.encoded"}),
        requires=CapabilitySet.of({"data.sample.omni"}),
    )


def test_parquet_row_groups_are_sharded_before_io_and_resume_exactly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.parquet"
    parquet.write_table(pa.Table.from_pylist(rows(8)), path, row_group_size=2)

    rank0_raw = build_parquet(path)
    rank1_raw = build_parquet(path)
    rank0_raw.shard(rank=0, world_size=2)
    rank1_raw.shard(rank=1, world_size=2)
    rank0 = AdaptedSource(rank0_raw, build_adapter())
    rank1 = AdaptedSource(rank1_raw, build_adapter())
    left = collect(rank0)
    right = collect(rank1)

    assert {sample.sample_id for sample in left}.isdisjoint(
        sample.sample_id for sample in right
    )
    assert {sample.sample_id for sample in left + right} == {
        f"row-{index}" for index in range(8)
    }
    assert rank0.metrics()["data/columnar/assigned_fragments"] == 2
    assert all(sample.messages[0].content[0].kind == "image" for sample in left)
    assert all(sample.metadata["quality"] >= 0 for sample in left)

    uninterrupted_raw = build_parquet(path, repeat=True)
    uninterrupted_raw.shard(rank=1, world_size=2)
    uninterrupted = AdaptedSource(uninterrupted_raw, build_adapter())
    uninterrupted.next_sample()
    state = uninterrupted.state_dict()
    expected = uninterrupted.next_sample().sample_id

    restored_raw = build_parquet(path, repeat=True)
    restored_raw.shard(rank=1, world_size=2)
    restored = AdaptedSource(restored_raw, build_adapter())
    restored.load_state_dict(state)
    assert restored.next_sample().sample_id == expected

    wrong_topology = AdaptedSource(build_parquet(path, repeat=True), build_adapter())
    with pytest.raises(CheckpointError, match="topology changed"):
        wrong_topology.load_state_dict(state)


def test_parquet_rank_and_worker_partitions_are_complete_and_disjoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.parquet"
    parquet.write_table(pa.Table.from_pylist(rows(16)), path, row_group_size=2)

    partitions = []
    for rank in range(2):
        for worker_id in range(2):
            source = AdaptedSource(build_parquet(path), build_adapter())
            source.shard(
                rank=rank,
                world_size=2,
                worker_id=worker_id,
                num_workers=2,
            )
            partitions.append({sample.sample_id for sample in collect(source)})

    assert set().union(*partitions) == {f"row-{index}" for index in range(16)}
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            assert left.isdisjoint(right)


def test_parquet_adapter_builds_through_the_complete_data_pipeline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.parquet"
    parquet.write_table(pa.Table.from_pylist(rows(4)), path, row_group_size=2)
    spec = DataPipelineSpec.from_mapping(
        {
            "source": {
                "module": "data_source:trainomni/parquet@1",
                "config": {
                    "dataset_id": "pipeline-fixture",
                    "paths": [str(path)],
                },
            },
            "adapter": {
                "module": "data_adapter:trainomni/msswift@1",
                "config": {"metadata_columns": ["quality"]},
            },
            "transforms": [],
            "model_io": {"module": "model_io:test/columnar@1"},
            "supervision": {"module": "supervision:trainomni/causal_lm@1"},
            "packer": {"module": "packer:trainomni/none@1"},
            "collator": {"module": "collator:trainomni/multimodal@1"},
        }
    )
    resolver = ModuleResolver(
        ModuleRegistry(
            (
                parquet_descriptor(),
                adapter_descriptor(),
                model_io_descriptor(),
                supervision_descriptor(),
                packer_descriptor(),
                collator_descriptor(),
            )
        )
    )

    stream = build_data_stream(spec, resolver, context=BuildContext("task"))
    batch = stream.next_batch(2)

    assert batch.sample_ids == ("row-0", "row-1")
    assert batch.model_inputs["input_ids"].shape == (2, 3)
    assert batch.model_inputs["pixel_values"].shape == (2, 12, 3)
    assert batch.labels.shape == (2, 3)


def test_arrow_file_and_stream_are_read_in_bounded_batches(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(rows(5))
    file_path = tmp_path / "file.arrow"
    with pa.OSFile(str(file_path), "wb") as sink, ipc.new_file(
        sink, table.schema
    ) as writer:
        for batch in table.to_batches(max_chunksize=2):
            writer.write_batch(batch)

    source = AdaptedSource(build_arrow(file_path), build_adapter())
    samples = collect(source)
    assert [sample.sample_id for sample in samples] == [
        f"row-{index}" for index in range(5)
    ]
    assert samples[0].messages[0].content[0].value.size == (4, 3)

    stream_path = tmp_path / "stream.arrow"
    with pa.OSFile(str(stream_path), "wb") as sink, ipc.new_stream(
        sink, table.schema
    ) as writer:
        for batch in table.to_batches(max_chunksize=2):
            writer.write_batch(batch)
    assert len(collect(AdaptedSource(build_arrow(stream_path), build_adapter()))) == 5


def test_msswift_adapter_fails_on_partial_media_placeholder_alignment(
    tmp_path: Path,
) -> None:
    raw = rows(1)
    raw[0]["images"].append(raw[0]["images"][0])
    path = tmp_path / "bad.parquet"
    parquet.write_table(pa.Table.from_pylist(raw), path)
    source = AdaptedSource(build_parquet(path), build_adapter())
    with pytest.raises(SpecError, match="placeholder count mismatch"):
        source.next_sample()
