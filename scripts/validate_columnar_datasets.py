"""Validate two real ms-swift-style datasets through Parquet and Arrow sources."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import pyarrow as pa
from pyarrow import ipc, parquet

from trainomni.core.context import BuildContext
from trainomni.core.module import ModuleRef
from trainomni.modules.data.adapters.binding import AdaptedSource
from trainomni.modules.data.adapters.msswift.module import (
    descriptor as adapter_descriptor,
)
from trainomni.modules.data.sources.arrow.module import (
    descriptor as arrow_descriptor,
)
from trainomni.modules.data.sources.parquet.module import (
    descriptor as parquet_descriptor,
)


def _reference(module: str, config: dict, *, field: str) -> ModuleRef:
    return ModuleRef.from_mapping(
        {"module": module, "config": config},
        field_name=field,
    )


def _adapter(context: BuildContext):
    return adapter_descriptor().build(
        _reference(
            "data_adapter:trainomni/msswift@1",
            {"decode_image_bytes": True},
            field="data.adapter",
        ),
        context,
    )


def _source(
    *,
    format_name: str,
    dataset_id: str,
    paths: list[str],
    context: BuildContext,
):
    descriptor = parquet_descriptor() if format_name == "parquet" else arrow_descriptor()
    return descriptor.build(
        _reference(
            f"data_source:trainomni/{format_name}@1",
            {
                "dataset_id": dataset_id,
                "paths": paths,
                "batch_rows": 64,
                "repeat": False,
            },
            field="data.source",
        ),
        context,
    )


def _convert_parquet_to_arrow_shards(source: Path, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    parquet_file = parquet.ParquetFile(source)
    paths = []
    for row_group in range(parquet_file.metadata.num_row_groups):
        batches = list(
            parquet_file.iter_batches(batch_size=64, row_groups=[row_group])
        )
        if not batches:
            continue
        path = output / f"part-{row_group:05d}.arrow"
        with pa.OSFile(str(path), "wb") as sink, ipc.new_file(
            sink, batches[0].schema
        ) as writer:
            for batch in batches:
                writer.write_batch(batch)
        paths.append(str(path))
    return paths


def _validate(
    *,
    format_name: str,
    dataset_id: str,
    paths: list[str],
    expected_rows: int,
    ranks: int,
) -> dict:
    context = BuildContext(task_digest=f"validation-{dataset_id}")
    started = time.perf_counter()
    rank_counts = []
    sample_ids = set()
    image_samples = 0
    for rank in range(ranks):
        raw_source = _source(
            format_name=format_name,
            dataset_id=dataset_id,
            paths=paths,
            context=context,
        )
        raw_source.shard(rank=rank, world_size=ranks)
        source = AdaptedSource(raw_source, _adapter(context))
        count = 0
        while True:
            try:
                sample = source.next_sample()
            except StopIteration:
                break
            if sample.sample_id in sample_ids:
                raise RuntimeError(f"duplicate sample id: {sample.sample_id}")
            sample_ids.add(sample.sample_id)
            if not sample.messages:
                raise RuntimeError(f"sample has no messages: {sample.sample_id}")
            if tuple(message.role for message in sample.messages[:2]) != (
                "user",
                "assistant",
            ):
                raise RuntimeError(f"unexpected roles: {sample.sample_id}")
            if any(
                block.kind == "image"
                for message in sample.messages
                for block in message.content
            ):
                image_samples += 1
            count += 1
        rank_counts.append(count)
    if len(sample_ids) != expected_rows:
        raise RuntimeError(
            f"{dataset_id}: expected {expected_rows} rows, got {len(sample_ids)}"
        )
    if image_samples != expected_rows:
        raise RuntimeError(
            f"{dataset_id}: expected one image-bearing sample per row, got "
            f"{image_samples}/{expected_rows}"
        )
    elapsed = time.perf_counter() - started
    return {
        "dataset_id": dataset_id,
        "format": format_name,
        "rows": expected_rows,
        "image_samples": image_samples,
        "ranks_simulated": ranks,
        "rows_per_rank": rank_counts,
        "duplicate_sample_ids": 0,
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(expected_rows / elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dataset", type=Path, required=True)
    parser.add_argument("--arrow-source-parquet", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    if args.ranks <= 0:
        parser.error("--ranks must be positive")
    for path in (args.parquet_dataset, args.arrow_source_parquet):
        if not path.is_file():
            parser.error(f"dataset does not exist: {path}")

    parquet_rows = parquet.ParquetFile(args.parquet_dataset).metadata.num_rows
    arrow_rows = parquet.ParquetFile(args.arrow_source_parquet).metadata.num_rows
    parquet_result = _validate(
        format_name="parquet",
        dataset_id="intergps-real",
        paths=[str(args.parquet_dataset.resolve())],
        expected_rows=parquet_rows,
        ranks=args.ranks,
    )
    with tempfile.TemporaryDirectory(prefix="trainomni-arrow-validation-") as temporary:
        arrow_paths = _convert_parquet_to_arrow_shards(
            args.arrow_source_parquet,
            Path(temporary),
        )
        arrow_result = _validate(
            format_name="arrow",
            dataset_id="diagram-image-to-text-real",
            paths=arrow_paths,
            expected_rows=arrow_rows,
            ranks=args.ranks,
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "datasets": [parquet_result, arrow_result],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
