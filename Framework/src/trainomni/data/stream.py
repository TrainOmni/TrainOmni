"""Compose a stateful physical reader and semantic importer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from trainomni.config import DatasetSpec

from .importers import ImportedSample, ImporterRegistry, import_with_trace
from .readers import ReaderRegistry, StatefulRecordReader

DATASET_STREAM_STATE_VERSION = "trainomni.dataset-stream.v1"


class DatasetStream:
    def __init__(
        self,
        spec: DatasetSpec,
        *,
        base_dir: Path | None,
        readers: ReaderRegistry | None = None,
        importers: ImporterRegistry | None = None,
    ) -> None:
        self.spec = spec
        self._readers = readers or ReaderRegistry()
        self._importers = importers or ImporterRegistry()
        reader_id = spec.config.get("reader")
        reader_config = spec.config.get("reader_config", {})
        importer_config = spec.config.get("importer", {})
        unknown = set(spec.config) - {"reader", "reader_config", "importer"}
        if unknown:
            raise ValueError(
                f"dataset {spec.dataset_id!r} has unknown config fields: {sorted(unknown)}"
            )
        if not isinstance(importer_config, Mapping):
            raise TypeError("dataset importer config must be a mapping")
        if not isinstance(reader_config, Mapping):
            raise TypeError("dataset reader_config must be a mapping")
        self._importer_config = importer_config
        self.reader: StatefulRecordReader = self._readers.open(
            spec.uri,
            base_dir=base_dir,
            reader_id=reader_id,
            config=reader_config,
        )
        self.importer = self._importers.get(spec.importer)
        self._initial_reader_state = self.reader.state_dict()

    def __iter__(self) -> Iterator[ImportedSample]:
        source_fingerprint = self.reader.fingerprint()
        for record in self.reader:
            yield import_with_trace(
                importer=self.importer,
                record=record,
                config=self._importer_config,
                dataset_id=self.spec.dataset_id,
                source_fingerprint=source_fingerprint,
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": DATASET_STREAM_STATE_VERSION,
            "dataset_id": self.spec.dataset_id,
            "reader": self.reader.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("state_version") != DATASET_STREAM_STATE_VERSION:
            raise ValueError("dataset stream state version mismatch")
        if state.get("dataset_id") != self.spec.dataset_id:
            raise ValueError("dataset stream ID mismatch")
        reader_state = state.get("reader")
        if not isinstance(reader_state, Mapping):
            raise TypeError("dataset stream reader state must be a mapping")
        self.reader.load_state_dict(reader_state)

    def reset(self) -> None:
        """Rewind after a complete epoch while retaining source identity checks."""

        self.reader.load_state_dict(self._initial_reader_state)


def open_dataset_streams(
    specs: tuple[DatasetSpec, ...],
    *,
    source_config: Path | None,
    readers: ReaderRegistry | None = None,
    importers: ImporterRegistry | None = None,
) -> tuple[DatasetStream, ...]:
    base_dir = source_config.parent if source_config is not None else None
    return tuple(
        DatasetStream(
            spec,
            base_dir=base_dir,
            readers=readers,
            importers=importers,
        )
        for spec in specs
    )
