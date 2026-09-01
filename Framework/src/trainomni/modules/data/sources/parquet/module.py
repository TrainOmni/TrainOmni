"""Row-group planned Parquet source with bounded Arrow batches."""

from __future__ import annotations

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .._columnar import (
    ColumnarRecordSource,
    PhysicalFragment,
    pyarrow_rows,
    resolve_local_files,
)
from .config import ParquetSourceConfig


def _require_parquet():
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise SpecError("Parquet data requires pyarrow>=15") from exc
    return parquet


def _factory(config: ParquetSourceConfig, context):
    parquet = _require_parquet()
    paths = resolve_local_files(
        config.paths,
        task_root=context.task_root,
        suffix=".parquet",
    )
    fragments = []
    for file_index, path in enumerate(paths):
        try:
            parquet_file = parquet.ParquetFile(path)
        except Exception as exc:
            raise SpecError(f"cannot open Parquet file {path}: {exc}") from exc
        schema_names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(config.columns) - schema_names)
        if missing:
            raise SpecError(
                f"Parquet file {path} is missing columns: {', '.join(missing)}"
            )
        stat = path.stat()
        for row_group in range(parquet_file.metadata.num_row_groups):
            rows = parquet_file.metadata.row_group(row_group).num_rows
            if rows == 0:
                continue
            fragments.append(
                PhysicalFragment(
                    path=path,
                    fragment_id=f"file-{file_index}:row-group-{row_group}",
                    rows=rows,
                    metadata={
                        "file_size": stat.st_size,
                        "file_mtime_ns": stat.st_mtime_ns,
                        "row_group": row_group,
                        "schema": str(parquet_file.schema_arrow),
                    },
                )
            )
    if not fragments:
        raise SpecError("Parquet dataset contains no non-empty row groups")

    def iter_fragment(fragment):
        row_group = int(fragment.metadata["row_group"])
        try:
            parquet_file = parquet.ParquetFile(fragment.path)
            batches = parquet_file.iter_batches(
                batch_size=config.batch_rows,
                row_groups=[row_group],
                columns=list(config.columns) or None,
                use_threads=True,
            )
            yield from pyarrow_rows(batches)
        except Exception as exc:
            raise SpecError(
                f"failed reading {fragment.path} row group {row_group}: {exc}"
            ) from exc

    return ColumnarRecordSource(
        dataset_id=config.dataset_id,
        fragments=fragments,
        iter_fragment=iter_fragment,
        repeat=config.repeat,
        format_name="parquet",
    )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("data_source:trainomni/parquet@1"),
        config_type=ParquetSourceConfig,
        factory=_factory,
        provides=CapabilitySet.of({"data.record.row", "data.source.stateful"}),
    )
