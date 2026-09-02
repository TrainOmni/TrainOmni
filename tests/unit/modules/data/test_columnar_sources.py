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
from trainomni.modules.data.packing.sequence.module import (
    descriptor as sequence_packer_descriptor,
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
from trainomni.runtime.data_loader import build_stateful_batch_loader
from trainomni.specs.run import DataLoaderSpec
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


def build_parquet(
    path: Path,
    *,
    repeat: bool = False,
    dataset_manifest_sha256: str | None = None,
):
    reference = ModuleRef.from_mapping(
        {
            "module": "data_source:trainomni/parquet@1",
            "config": {
                "dataset_id": "parquet-fixture",
                "paths": [str(path)],
                "batch_rows": 2,
                "repeat": repeat,
                "dataset_manifest_sha256": dataset_manifest_sha256,
            },
        },
        field_name="data.source",
    )
    return parquet_descriptor().build(reference, BuildContext("task"))


def build_arrow(
    path: Path,
    *,
    repeat: bool = False,
    dataset_manifest_sha256: str | None = None,
):
    reference = ModuleRef.from_mapping(
        {
            "module": "data_source:trainomni/arrow@1",
            "config": {
                "dataset_id": "arrow-fixture",
                "paths": [str(path)],
                "batch_rows": 2,
                "repeat": repeat,
                "dataset_manifest_sha256": dataset_manifest_sha256,
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


def test_repeating_columnar_shards_require_equal_rows_and_resume(tmp_path: Path) -> None:
    uneven_path = tmp_path / "uneven.parquet"
    parquet.write_table(pa.Table.from_pylist(rows(5)), uneven_path, row_group_size=4)
    with pytest.raises(SpecError, match="equal assigned row totals.*4, 1"):
        build_parquet(uneven_path, repeat=True).shard(rank=0, world_size=2)

    balanced_path = tmp_path / "balanced.parquet"
    parquet.write_table(pa.Table.from_pylist(rows(4)), balanced_path, row_group_size=2)
    rank0 = build_parquet(balanced_path, repeat=True)
    rank1 = build_parquet(balanced_path, repeat=True)
    rank0.shard(rank=0, world_size=2)
    rank1.shard(rank=1, world_size=2)
    rank0_ids = [rank0.next_record().sample_id for _ in range(4)]
    rank1_ids = [rank1.next_record().sample_id for _ in range(4)]
    assert len(set(rank0_ids)) == len(set(rank1_ids)) == 2

    state = rank1.state_dict()
    expected = [rank1.next_record().sample_id for _ in range(4)]
    restored = build_parquet(balanced_path, repeat=True)
    restored.shard(rank=1, world_size=2)
    restored.load_state_dict(state)
    assert [restored.next_record().sample_id for _ in range(4)] == expected


def test_repeating_columnar_rejects_intergps_shaped_680_600_split(
    tmp_path: Path,
) -> None:
    path = tmp_path / "intergps-layout.parquet"
    parquet.write_table(
        pa.Table.from_pylist([{"id": index} for index in range(1280)]),
        path,
        row_group_size=100,
    )
    with pytest.raises(SpecError, match=r"\(680, 600\)"):
        build_parquet(path, repeat=True).shard(rank=0, world_size=2)


def test_columnar_resume_uses_semantic_manifest_not_physical_root(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original" / "samples.parquet"
    original.parent.mkdir()
    parquet.write_table(pa.Table.from_pylist(rows(4)), original, row_group_size=2)
    manifest = "d" * 64

    source = build_parquet(
        original,
        repeat=True,
        dataset_manifest_sha256=manifest,
    )
    source.next_record()
    state = source.state_dict()
    expected = source.next_record()

    moved = tmp_path / "relocated" / original.name
    moved.parent.mkdir()
    moved.write_bytes(original.read_bytes())
    restored = build_parquet(
        moved,
        repeat=True,
        dataset_manifest_sha256=manifest,
    )
    restored.load_state_dict(state)
    actual = restored.next_record()
    assert actual.fields["id"] == expected.fields["id"]
    assert actual.sample_id == expected.sample_id
    assert actual.position == expected.position
    assert restored.state_dict()["identity"] == state["identity"]

    changed_snapshot = build_parquet(
        moved,
        repeat=True,
        dataset_manifest_sha256="e" * 64,
    )
    with pytest.raises(CheckpointError, match="dataset identity changed"):
        changed_snapshot.load_state_dict(state)


def test_columnar_restore_rejects_unreachable_emitted_count(tmp_path: Path) -> None:
    path = tmp_path / "samples.parquet"
    parquet.write_table(pa.Table.from_pylist(rows(4)), path, row_group_size=2)

    source = build_parquet(path, repeat=True)
    state = dict(source.state_dict())
    state["emitted"] = 99

    with pytest.raises(CheckpointError, match="emitted count is inconsistent"):
        build_parquet(path, repeat=True).load_state_dict(state)


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


@pytest.mark.parametrize("format_name", ["parquet", "arrow"])
def test_columnar_pipeline_runs_in_stateful_multiworker_loader(
    tmp_path: Path,
    format_name: str,
) -> None:
    table = pa.Table.from_pylist(rows(8))
    path = tmp_path / f"samples.{format_name}"
    if format_name == "parquet":
        parquet.write_table(table, path, row_group_size=2)
        source_descriptor = parquet_descriptor()
    else:
        with pa.OSFile(str(path), "wb") as sink, ipc.new_file(
            sink, table.schema
        ) as writer:
            for batch in table.to_batches(max_chunksize=2):
                writer.write_batch(batch)
        source_descriptor = arrow_descriptor()
    spec = DataPipelineSpec.from_mapping(
        {
            "source": {
                "module": f"data_source:trainomni/{format_name}@1",
                "config": {
                    "dataset_id": f"{format_name}-workers",
                    "paths": [str(path)],
                    "batch_rows": 2,
                    "repeat": False,
                },
            },
            "adapter": {"module": "data_adapter:trainomni/msswift@1"},
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
                source_descriptor,
                adapter_descriptor(),
                model_io_descriptor(),
                supervision_descriptor(),
                packer_descriptor(),
                collator_descriptor(),
            )
        )
    )
    pipeline = build_data_stream(spec, resolver, context=BuildContext("task"))
    loader = build_stateful_batch_loader(
        pipeline,
        batch_size=2,
        spec=DataLoaderSpec(
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=False,
        ),
    )

    first_batch = loader.next_batch(2)
    state = loader.state_dict()
    remaining_ids = []
    while True:
        try:
            remaining_ids.extend(loader.next_batch(2).sample_ids)
        except StopIteration:
            break

    sample_ids = [*first_batch.sample_ids, *remaining_ids]
    assert len(sample_ids) == 8
    assert set(sample_ids) == {f"row-{index}" for index in range(8)}
    assert loader.metrics()["data/loader/batches"] == 4
    loader.close()

    restored_pipeline = build_data_stream(
        spec,
        resolver,
        context=BuildContext("task"),
    )
    restored = build_stateful_batch_loader(
        restored_pipeline,
        batch_size=2,
        spec=DataLoaderSpec(
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=False,
        ),
    )
    restored.load_state_dict(state)
    restored_ids = []
    while True:
        try:
            restored_ids.extend(restored.next_batch(2).sample_ids)
        except StopIteration:
            break
    restored.close()
    assert restored_ids == remaining_ids


@pytest.mark.parametrize("format_name", ["parquet", "arrow"])
def test_multiworker_columnar_sequence_packing_resumes_exactly(
    tmp_path: Path,
    format_name: str,
) -> None:
    table = pa.Table.from_pylist(rows(8))
    path = tmp_path / f"packed.{format_name}"
    if format_name == "parquet":
        parquet.write_table(table, path, row_group_size=2)
        source_descriptor = parquet_descriptor()
    else:
        with pa.OSFile(str(path), "wb") as sink, ipc.new_file(
            sink, table.schema
        ) as writer:
            for batch in table.to_batches(max_chunksize=2):
                writer.write_batch(batch)
        source_descriptor = arrow_descriptor()
    spec = DataPipelineSpec.from_mapping(
        {
            "source": {
                "module": f"data_source:trainomni/{format_name}@1",
                "config": {
                    "dataset_id": f"{format_name}-packed-workers",
                    "paths": [str(path)],
                    "batch_rows": 2,
                    "repeat": False,
                },
            },
            "adapter": {"module": "data_adapter:trainomni/msswift@1"},
            "transforms": [],
            "model_io": {"module": "model_io:test/columnar@1"},
            "supervision": {"module": "supervision:trainomni/causal_lm@1"},
            "packer": {
                "module": "packer:trainomni/sequence@1",
                "config": {
                    "max_length": 6,
                    "pad_token_id": 0,
                    "max_samples_per_pack": 2,
                    "concat_fields": ["pixel_values"],
                },
            },
            "collator": {"module": "collator:trainomni/multimodal@1"},
        }
    )
    resolver = ModuleResolver(
        ModuleRegistry(
            (
                source_descriptor,
                adapter_descriptor(),
                model_io_descriptor(),
                supervision_descriptor(),
                sequence_packer_descriptor(),
                collator_descriptor(),
            )
        )
    )

    def loader():
        return build_stateful_batch_loader(
            build_data_stream(spec, resolver, context=BuildContext("task")),
            batch_size=1,
            spec=DataLoaderSpec(
                num_workers=2,
                prefetch_factor=2,
                persistent_workers=False,
            ),
        )

    first = loader()
    initial = first.next_batch(1)
    state = first.state_dict()
    expected = []
    while True:
        try:
            batch = first.next_batch(1)
        except StopIteration:
            break
        expected.append(
            (
                batch.sample_ids,
                batch.model_inputs["input_ids"].clone(),
                batch.model_inputs["packed_attention_mask"].clone(),
            )
        )
    first.close()
    assert initial.supervision["packed_lengths"].tolist() == [[3, 3]]
    assert len(expected) == 3

    restored = loader()
    restored.load_state_dict(state)
    actual = []
    while True:
        try:
            batch = restored.next_batch(1)
        except StopIteration:
            break
        actual.append(
            (
                batch.sample_ids,
                batch.model_inputs["input_ids"].clone(),
                batch.model_inputs["packed_attention_mask"].clone(),
            )
        )
    restored.close()
    assert [item[0] for item in actual] == [item[0] for item in expected]
    for actual_item, expected_item in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_item[1], expected_item[1])
        torch.testing.assert_close(actual_item[2], expected_item[2])


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
