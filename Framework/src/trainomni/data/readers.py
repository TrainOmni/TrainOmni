"""Stateful physical record readers.

Readers know storage, never VLM semantics. Importers convert a raw record into
the canonical sample contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

READER_STATE_VERSION = "trainomni.reader-state.v1"


class DataReadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RawRecord:
    value: Mapping[str, Any]
    source_uri: str
    record_index: int
    byte_offset: int | None = None


class StatefulRecordReader(Protocol):
    reader_id: str

    def __iter__(self) -> Iterator[RawRecord]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def fingerprint(self) -> str: ...


def resolve_local_uri(uri: str, *, base_dir: Path | None = None) -> Path:
    if re.match(r"^[A-Za-z]:[\\/]", uri):
        return Path(uri).resolve()
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise DataReadError(
            f"reader only supports local/file URIs in M1, got scheme {parsed.scheme!r}"
        )
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise DataReadError(f"remote file URI host is unsupported: {parsed.netloc!r}")
        path = Path(unquote(parsed.path))
        # file:///D:/... is parsed with a leading slash on Windows.
        if len(path.as_posix()) >= 4 and path.as_posix()[0] == "/" and path.as_posix()[2] == ":":
            path = Path(path.as_posix()[1:])
    else:
        path = Path(uri)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonlReader:
    reader_id = "jsonl"

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise DataReadError(f"JSONL dataset does not exist: {self.path}")
        self._next_index = 0
        self._next_offset = 0
        self._fingerprint: str | None = None

    def __iter__(self) -> JsonlReader:
        return self

    def __next__(self) -> RawRecord:
        with self.path.open("rb") as handle:
            handle.seek(self._next_offset)
            byte_offset = handle.tell()
            raw_line = handle.readline()
            next_offset = handle.tell()
        if not raw_line:
            raise StopIteration
        index = self._next_index
        stripped = raw_line.strip()
        if not stripped:
            raise DataReadError(f"blank JSONL record at {self.path}:{index + 1}")
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataReadError(
                f"invalid JSON at {self.path}:{index + 1}: {exc.msg}"
            ) from exc
        if not isinstance(value, Mapping):
            raise DataReadError(
                f"record at {self.path}:{index + 1} must be an object"
            )
        self._next_index = index + 1
        self._next_offset = next_offset
        return RawRecord(
            value=value,
            source_uri=str(self.path),
            record_index=index,
            byte_offset=byte_offset,
        )

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = file_fingerprint(self.path)
        return self._fingerprint

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": READER_STATE_VERSION,
            "reader_id": self.reader_id,
            "path": str(self.path),
            "fingerprint": self.fingerprint(),
            "next_index": self._next_index,
            "next_offset": self._next_offset,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _validate_reader_state(state, self.reader_id, self.path, self.fingerprint())
        next_index = state.get("next_index")
        next_offset = state.get("next_offset")
        if not isinstance(next_index, int) or next_index < 0:
            raise DataReadError("reader state next_index must be a non-negative integer")
        if (
            not isinstance(next_offset, int)
            or next_offset < 0
            or next_offset > self.path.stat().st_size
        ):
            raise DataReadError("reader state next_offset is outside the dataset")
        self._next_index = next_index
        self._next_offset = next_offset


class JsonArrayReader:
    reader_id = "json_array"

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise DataReadError(f"JSON dataset does not exist: {self.path}")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataReadError(f"cannot parse JSON dataset {self.path}: {exc}") from exc
        if not isinstance(value, list):
            raise DataReadError(f"JSON dataset root must be an array: {self.path}")
        if not all(isinstance(item, Mapping) for item in value):
            raise DataReadError(f"every JSON dataset record must be an object: {self.path}")
        self._records = value
        self._next_index = 0
        self._fingerprint = file_fingerprint(self.path)

    def __iter__(self) -> Iterator[RawRecord]:
        while self._next_index < len(self._records):
            index = self._next_index
            self._next_index += 1
            yield RawRecord(
                value=self._records[index],
                source_uri=str(self.path),
                record_index=index,
            )

    def fingerprint(self) -> str:
        return self._fingerprint

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": READER_STATE_VERSION,
            "reader_id": self.reader_id,
            "path": str(self.path),
            "fingerprint": self.fingerprint(),
            "next_index": self._next_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _validate_reader_state(state, self.reader_id, self.path, self.fingerprint())
        next_index = state.get("next_index")
        if not isinstance(next_index, int) or not 0 <= next_index <= len(self._records):
            raise DataReadError("reader state next_index is outside the dataset")
        self._next_index = next_index


class ParquetReader:
    reader_id = "parquet"

    def __init__(self, path: Path, *, columns: tuple[str, ...] | None = None) -> None:
        try:
            from pyarrow import parquet
        except ImportError as exc:
            raise DataReadError(
                "parquet reader requires trainomni-framework[data]"
            ) from exc
        self.path = path.resolve()
        if not self.path.is_file():
            raise DataReadError(f"Parquet dataset does not exist: {self.path}")
        self._parquet = parquet
        self._columns = columns
        source = parquet.ParquetFile(self.path)
        try:
            self._num_row_groups = source.num_row_groups
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()
        self._row_group = 0
        self._row_index = 0
        self._record_index = 0
        self._table: Any | None = None
        self._fingerprint = file_fingerprint(self.path)

    def __iter__(self) -> ParquetReader:
        return self

    def __next__(self) -> RawRecord:
        while self._row_group < self._num_row_groups:
            if self._table is None:
                source = self._parquet.ParquetFile(self.path)
                try:
                    self._table = source.read_row_group(
                        self._row_group, columns=self._columns
                    )
                finally:
                    close = getattr(source, "close", None)
                    if callable(close):
                        close()
            if self._row_index < self._table.num_rows:
                index = self._record_index
                value = self._table.slice(self._row_index, 1).to_pylist()[0]
                self._row_index += 1
                self._record_index += 1
                return RawRecord(value, str(self.path), index)
            self._row_group += 1
            self._row_index = 0
            self._table = None
        raise StopIteration

    def fingerprint(self) -> str:
        return self._fingerprint

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": READER_STATE_VERSION,
            "reader_id": self.reader_id,
            "path": str(self.path),
            "fingerprint": self.fingerprint(),
            "columns": self._columns,
            "row_group": self._row_group,
            "row_index": self._row_index,
            "record_index": self._record_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _validate_reader_state(state, self.reader_id, self.path, self.fingerprint())
        if state.get("columns") != self._columns:
            raise DataReadError("Parquet reader column projection mismatch")
        values = [state.get(key) for key in ("row_group", "row_index", "record_index")]
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise DataReadError("Parquet reader position is invalid")
        row_group, row_index, _record_index = values
        if row_group > self._num_row_groups:
            raise DataReadError("Parquet row group is outside the dataset")
        if row_group == self._num_row_groups and row_index != 0:
            raise DataReadError("Parquet terminal state has a non-zero row index")
        if row_group < self._num_row_groups:
            source = self._parquet.ParquetFile(self.path)
            try:
                row_count = source.metadata.row_group(row_group).num_rows
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()
            if row_index > row_count:
                raise DataReadError("Parquet row index is outside the row group")
        self._row_group, self._row_index, self._record_index = values
        self._table = None


class TarJsonReader:
    reader_id = "tar_json"

    def __init__(self, path: Path, *, suffix: str = ".json") -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise DataReadError(f"tar dataset does not exist: {self.path}")
        self._suffix = suffix
        try:
            with tarfile.open(self.path, "r:*") as archive:
                self._members = tuple(
                    member.name
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith(suffix)
                )
        except tarfile.TarError as exc:
            raise DataReadError(f"cannot open tar dataset {self.path}") from exc
        self._next_index = 0
        self._fingerprint = file_fingerprint(self.path)

    def __iter__(self) -> TarJsonReader:
        return self

    def __next__(self) -> RawRecord:
        if self._next_index >= len(self._members):
            raise StopIteration
        index = self._next_index
        name = self._members[index]
        try:
            with tarfile.open(self.path, "r:*") as archive:
                handle = archive.extractfile(name)
                if handle is None:
                    raise DataReadError(f"tar member disappeared: {name}")
                value = json.load(handle)
        except (tarfile.TarError, json.JSONDecodeError) as exc:
            raise DataReadError(f"invalid JSON tar member {name!r}") from exc
        if not isinstance(value, Mapping):
            raise DataReadError(f"tar member {name!r} must contain a JSON object")
        self._next_index += 1
        return RawRecord(value, f"tar://{self.path}!/{name}", index)

    def fingerprint(self) -> str:
        return self._fingerprint

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": READER_STATE_VERSION,
            "reader_id": self.reader_id,
            "path": str(self.path),
            "fingerprint": self.fingerprint(),
            "suffix": self._suffix,
            "members": self._members,
            "next_index": self._next_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _validate_reader_state(state, self.reader_id, self.path, self.fingerprint())
        if state.get("suffix") != self._suffix or tuple(state.get("members", ())) != self._members:
            raise DataReadError("tar reader member index mismatch")
        next_index = state.get("next_index")
        if not isinstance(next_index, int) or not 0 <= next_index <= len(self._members):
            raise DataReadError("tar reader position is invalid")
        self._next_index = next_index


def _validate_reader_state(
    state: Mapping[str, Any], reader_id: str, path: Path, fingerprint: str
) -> None:
    expected = {
        "state_version": READER_STATE_VERSION,
        "reader_id": reader_id,
        "path": str(path),
        "fingerprint": fingerprint,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise DataReadError(
                f"reader state {key} mismatch: expected {value!r}, got {state.get(key)!r}"
            )


class ReaderRegistry:
    """Select built-in physical readers without importing dataset semantics."""

    def __init__(self) -> None:
        self._factories: dict[
            str, Callable[[Path, Mapping[str, Any]], StatefulRecordReader]
        ] = {}
        self.register("jsonl", _jsonl_factory)
        self.register("json_array", _json_array_factory)
        self.register("parquet", _parquet_factory)
        self.register("tar_json", _tar_json_factory)

    def register(
        self,
        reader_id: str,
        factory: Callable[[Path, Mapping[str, Any]], StatefulRecordReader],
    ) -> None:
        if not reader_id.strip() or not callable(factory):
            raise DataReadError("reader registration requires an ID and factory")
        if reader_id in self._factories:
            raise DataReadError(f"reader {reader_id!r} is already registered")
        self._factories[reader_id] = factory

    def open(
        self,
        uri: str,
        *,
        base_dir: Path | None = None,
        reader_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> StatefulRecordReader:
        path = resolve_local_uri(uri, base_dir=base_dir)
        selected = reader_id
        if selected is None:
            suffix = path.suffix.lower()
            selected = {
                ".jsonl": "jsonl",
                ".parquet": "parquet",
                ".tar": "tar_json",
                ".tgz": "tar_json",
            }.get(suffix, "json_array")
        try:
            factory = self._factories[selected]
        except KeyError as exc:
            raise DataReadError(
                f"unknown reader {selected!r}; available: {sorted(self._factories)}"
            ) from exc
        return factory(path, dict(config or {}))

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _jsonl_factory(path: Path, config: Mapping[str, Any]) -> StatefulRecordReader:
    if config:
        raise DataReadError(f"jsonl reader has unknown config: {sorted(config)}")
    return JsonlReader(path)


def _json_array_factory(path: Path, config: Mapping[str, Any]) -> StatefulRecordReader:
    if config:
        raise DataReadError(f"json_array reader has unknown config: {sorted(config)}")
    return JsonArrayReader(path)


def _parquet_factory(path: Path, config: Mapping[str, Any]) -> StatefulRecordReader:
    unknown = set(config) - {"columns"}
    if unknown:
        raise DataReadError(f"parquet reader has unknown config: {sorted(unknown)}")
    columns = config.get("columns")
    if columns is not None and (
        not isinstance(columns, (list, tuple))
        or not all(isinstance(item, str) and item for item in columns)
    ):
        raise DataReadError("parquet columns must be a list of names")
    return ParquetReader(path, columns=tuple(columns) if columns else None)


def _tar_json_factory(path: Path, config: Mapping[str, Any]) -> StatefulRecordReader:
    unknown = set(config) - {"suffix"}
    if unknown:
        raise DataReadError(f"tar_json reader has unknown config: {sorted(unknown)}")
    suffix = config.get("suffix", ".json")
    if not isinstance(suffix, str) or not suffix:
        raise DataReadError("tar_json suffix must be a non-empty string")
    return TarJsonReader(path, suffix=suffix)
