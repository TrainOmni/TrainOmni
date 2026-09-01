"""Arrow IPC file/stream source with physical file or record-batch planning."""

from __future__ import annotations

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .._columnar import ColumnarRecordSource, PhysicalFragment, resolve_local_files
from .config import ArrowSourceConfig


def _require_arrow():
    try:
        import pyarrow as arrow
        from pyarrow import ipc
    except ImportError as exc:
        raise SpecError("Arrow IPC data requires pyarrow>=15") from exc
    return arrow, ipc


def _project(batch, columns):
    if not columns:
        return batch
    return batch.select(list(columns))


def _batch_rows(batch, *, batch_rows: int):
    for offset in range(0, batch.num_rows, batch_rows):
        yield from batch.slice(offset, batch_rows).to_pylist()


def _factory(config: ArrowSourceConfig, context):
    arrow, ipc = _require_arrow()
    paths = resolve_local_files(
        config.paths,
        task_root=context.task_root,
        suffix=".arrow",
    )
    fragments = []
    for file_index, path in enumerate(paths):
        stat = path.stat()
        source = arrow.memory_map(str(path), "r")
        try:
            try:
                reader = ipc.open_file(source)
                schema = reader.schema
                missing = sorted(set(config.columns) - set(schema.names))
                if missing:
                    raise SpecError(
                        f"Arrow file {path} is missing columns: {', '.join(missing)}"
                    )
                for batch_index in range(reader.num_record_batches):
                    rows = reader.get_batch(batch_index).num_rows
                    if rows == 0:
                        continue
                    fragments.append(
                        PhysicalFragment(
                            path=path,
                            fragment_id=f"file-{file_index}:batch-{batch_index}",
                            rows=rows,
                            metadata={
                                "file_size": stat.st_size,
                                "file_mtime_ns": stat.st_mtime_ns,
                                "ipc_kind": "file",
                                "batch": batch_index,
                                "schema": str(schema),
                            },
                        )
                    )
            except arrow.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_stream(source)
                schema = reader.schema
                missing = sorted(set(config.columns) - set(schema.names))
                if missing:
                    raise SpecError(
                        f"Arrow stream {path} is missing columns: {', '.join(missing)}"
                    )
                rows = sum(batch.num_rows for batch in reader)
                if rows:
                    fragments.append(
                        PhysicalFragment(
                            path=path,
                            fragment_id=f"file-{file_index}:stream",
                            rows=rows,
                            metadata={
                                "file_size": stat.st_size,
                                "file_mtime_ns": stat.st_mtime_ns,
                                "ipc_kind": "stream",
                                "schema": str(schema),
                            },
                        )
                    )
        except SpecError:
            raise
        except Exception as exc:
            raise SpecError(f"cannot inspect Arrow IPC file {path}: {exc}") from exc
        finally:
            source.close()
    if not fragments:
        raise SpecError("Arrow dataset contains no non-empty physical fragments")

    def iter_fragment(fragment):
        source = arrow.memory_map(str(fragment.path), "r")
        try:
            if fragment.metadata["ipc_kind"] == "file":
                reader = ipc.open_file(source)
                batch = reader.get_batch(int(fragment.metadata["batch"]))
                yield from _batch_rows(
                    _project(batch, config.columns),
                    batch_rows=config.batch_rows,
                )
            else:
                reader = ipc.open_stream(source)
                for batch in reader:
                    yield from _batch_rows(
                        _project(batch, config.columns),
                        batch_rows=config.batch_rows,
                    )
        except Exception as exc:
            raise SpecError(f"failed reading Arrow IPC file {fragment.path}: {exc}") from exc
        finally:
            source.close()

    return ColumnarRecordSource(
        dataset_id=config.dataset_id,
        fragments=fragments,
        iter_fragment=iter_fragment,
        repeat=config.repeat,
        format_name="arrow",
    )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("data_source:trainomni/arrow@1"),
        config_type=ArrowSourceConfig,
        factory=_factory,
        provides=CapabilitySet.of({"data.record.row", "data.source.stateful"}),
    )
