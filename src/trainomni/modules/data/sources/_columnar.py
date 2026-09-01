"""Shared physical-fragment cursor for builtin columnar sources."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainomni.contracts.data import DataRecord
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.specs.digest import identity_digest


@dataclass(frozen=True, slots=True)
class PhysicalFragment:
    path: Path
    fragment_id: str
    rows: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("columnar fragments must contain at least one row")


def resolve_local_files(
    patterns: Sequence[str],
    *,
    task_root: Path | None,
    suffix: str,
) -> tuple[Path, ...]:
    import glob

    resolved: list[Path] = []
    for raw_pattern in patterns:
        candidate = Path(raw_pattern)
        if not candidate.is_absolute():
            if task_root is None:
                raise SpecError("relative columnar paths require a task root")
            candidate = task_root / candidate
        text = str(candidate)
        if glob.has_magic(text):
            matches = [Path(item) for item in glob.glob(text, recursive=True)]
        elif candidate.is_dir():
            matches = list(candidate.rglob(f"*{suffix}"))
        else:
            matches = [candidate]
        files = [item.resolve() for item in matches if item.is_file()]
        if not files:
            raise SpecError(f"columnar path matched no files: {raw_pattern!r}")
        invalid = [item for item in files if item.suffix.lower() != suffix]
        if invalid:
            raise SpecError(
                f"columnar path contains non-{suffix} file: {invalid[0]}"
            )
        resolved.extend(files)
    unique = tuple(sorted(set(resolved), key=lambda item: item.as_posix()))
    if not unique:
        raise SpecError("columnar source resolved no files")
    return unique


def balanced_fragment_assignment(
    fragments: Sequence[PhysicalFragment],
    *,
    rank: int,
    world_size: int,
) -> tuple[PhysicalFragment, ...]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise SpecError("columnar sharding requires an in-range rank")
    if len(fragments) < world_size:
        raise SpecError(
            "columnar source has fewer physical fragments than ranks; "
            "rewrite the dataset with more Parquet row groups or Arrow shards"
        )
    buckets: list[list[tuple[int, PhysicalFragment]]] = [
        [] for _ in range(world_size)
    ]
    loads = [0] * world_size
    ordered = sorted(
        enumerate(fragments),
        key=lambda item: (-item[1].rows, item[0]),
    )
    for original_index, fragment in ordered:
        target = min(range(world_size), key=lambda item: (loads[item], item))
        buckets[target].append((original_index, fragment))
        loads[target] += fragment.rows
    return tuple(fragment for _, fragment in sorted(buckets[rank]))


class ColumnarRecordSource:
    """Stateful iteration over rank-assigned physical fragments."""

    def __init__(
        self,
        *,
        dataset_id: str,
        fragments: Sequence[PhysicalFragment],
        iter_fragment: Callable[[PhysicalFragment], Iterable[Mapping[str, Any]]],
        repeat: bool,
        format_name: str,
        dataset_manifest_sha256: str | None,
    ) -> None:
        self.dataset_id = dataset_id
        self.fragments = tuple(fragments)
        self._iter_fragment = iter_fragment
        self.repeat = repeat
        self.is_finite = not repeat
        self.format_name = format_name
        self.identity = identity_digest(
            {
                "schema_version": 2,
                "dataset_id": dataset_id,
                "format": format_name,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "fragments": [
                    {
                        "fragment_id": fragment.fragment_id,
                        "rows": fragment.rows,
                        "metadata": {
                            key: value
                            for key, value in fragment.metadata.items()
                            if key not in {"file_size", "file_mtime_ns"}
                        },
                    }
                    for fragment in self.fragments
                ],
            }
        )
        self.rank = 0
        self.world_size = 1
        self.worker_id = 0
        self.num_workers = 1
        self.assigned = self.fragments
        self.epoch = 0
        self.fragment_cursor = 0
        self.row_cursor = 0
        self.emitted = 0
        self._active_iterator = None

    def shard(
        self,
        *,
        rank: int,
        world_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> None:
        if self.emitted or self.fragment_cursor or self.row_cursor:
            raise CheckpointError("columnar source must be sharded before reading")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise SpecError("columnar sharding requires an in-range rank")
        if num_workers <= 0 or not 0 <= worker_id < num_workers:
            raise SpecError("columnar sharding requires an in-range worker")
        requested = (rank, world_size, worker_id, num_workers)
        current = (self.rank, self.world_size, self.worker_id, self.num_workers)
        if current != (0, 1, 0, 1):
            if current != requested:
                raise CheckpointError("columnar source was already sharded")
            return
        partition = rank * num_workers + worker_id
        partitions = world_size * num_workers
        self.assigned = balanced_fragment_assignment(
            self.fragments,
            rank=partition,
            world_size=partitions,
        )
        self.rank = rank
        self.world_size = world_size
        self.worker_id = worker_id
        self.num_workers = num_workers

    def _activate(self) -> None:
        fragment = self.assigned[self.fragment_cursor]
        iterator = iter(self._iter_fragment(fragment))
        skipped = 0
        while skipped < self.row_cursor:
            try:
                next(iterator)
            except StopIteration as exc:
                raise CheckpointError(
                    "columnar cursor exceeds the physical fragment"
                ) from exc
            skipped += 1
        self._active_iterator = iterator

    def next_record(self) -> DataRecord:
        while True:
            if self.fragment_cursor >= len(self.assigned):
                if not self.repeat:
                    raise StopIteration
                self.epoch += 1
                self.fragment_cursor = 0
                self.row_cursor = 0
                self._active_iterator = None
            if self._active_iterator is None:
                self._activate()
            fragment = self.assigned[self.fragment_cursor]
            try:
                fields = next(self._active_iterator)
            except StopIteration:
                if self.row_cursor != fragment.rows:
                    raise SpecError(
                        f"fragment row count changed: {fragment.path}#"
                        f"{fragment.fragment_id} expected {fragment.rows}, "
                        f"read {self.row_cursor}"
                    )
                self.fragment_cursor += 1
                self.row_cursor = 0
                self._active_iterator = None
                continue
            row_index = self.row_cursor
            self.row_cursor += 1
            self.emitted += 1
            sample_id = f"{self.dataset_id}::{fragment.fragment_id}::{row_index}"
            return DataRecord(
                sample_id=sample_id,
                fields=fields,
                source=self.dataset_id,
                position={
                    "format": self.format_name,
                    "fragment": fragment.fragment_id,
                    "row": row_index,
                    "epoch": self.epoch,
                },
            )

    def state_dict(self):
        return {
            "identity": self.identity,
            "rank": self.rank,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "assigned_fragments": tuple(item.fragment_id for item in self.assigned),
            "epoch": self.epoch,
            "fragment_cursor": self.fragment_cursor,
            "row_cursor": self.row_cursor,
            "emitted": self.emitted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "identity",
            "rank",
            "world_size",
            "worker_id",
            "num_workers",
            "assigned_fragments",
            "epoch",
            "fragment_cursor",
            "row_cursor",
            "emitted",
        }
        if set(state) != expected:
            raise CheckpointError("invalid columnar source state keys")
        if state["identity"] != self.identity:
            raise CheckpointError("columnar dataset identity changed")
        if (
            int(state["rank"]),
            int(state["world_size"]),
            int(state["worker_id"]),
            int(state["num_workers"]),
        ) != (
            self.rank,
            self.world_size,
            self.worker_id,
            self.num_workers,
        ):
            raise CheckpointError("columnar source topology changed")
        assigned = tuple(str(item) for item in state["assigned_fragments"])
        if assigned != tuple(item.fragment_id for item in self.assigned):
            raise CheckpointError("columnar physical fragment assignment changed")
        epoch = int(state["epoch"])
        fragment_cursor = int(state["fragment_cursor"])
        row_cursor = int(state["row_cursor"])
        emitted = int(state["emitted"])
        if min(epoch, fragment_cursor, row_cursor, emitted) < 0:
            raise CheckpointError("columnar cursor values must be non-negative")
        if fragment_cursor > len(self.assigned):
            raise CheckpointError("columnar fragment cursor is out of range")
        if fragment_cursor == len(self.assigned) and row_cursor:
            raise CheckpointError("finished columnar cursor cannot have a row offset")
        if (
            fragment_cursor < len(self.assigned)
            and row_cursor > self.assigned[fragment_cursor].rows
        ):
            raise CheckpointError("columnar row cursor is out of range")
        self.epoch = epoch
        self.fragment_cursor = fragment_cursor
        self.row_cursor = row_cursor
        self.emitted = emitted
        self._active_iterator = None

    def metrics(self):
        return {
            "data/columnar/emitted": self.emitted,
            "data/columnar/epoch": self.epoch,
            "data/columnar/physical_fragments": len(self.fragments),
            "data/columnar/assigned_fragments": len(self.assigned),
            "data/columnar/assigned_rows": sum(item.rows for item in self.assigned),
            "data/columnar/worker_id": self.worker_id,
            "data/columnar/num_workers": self.num_workers,
        }


def pyarrow_rows(batches: Iterable[Any]) -> Iterable[Mapping[str, Any]]:
    for batch in batches:
        yield from batch.to_pylist()
