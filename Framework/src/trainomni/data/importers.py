"""Raw-record to canonical-sample importers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .canonical import canonical_hash, parse_sample
from .model import CanonicalSample
from .readers import RawRecord

IMPORTER_API_VERSION = "trainomni.importer.v1"


class DataImportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ImportTrace:
    dataset_id: str
    source_uri: str
    source_fingerprint: str
    record_index: int
    byte_offset: int | None
    importer_id: str
    importer_version: str
    sample_id: str
    sample_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_uri": self.source_uri,
            "source_fingerprint": self.source_fingerprint,
            "record_index": self.record_index,
            "byte_offset": self.byte_offset,
            "importer_id": self.importer_id,
            "importer_version": self.importer_version,
            "sample_id": self.sample_id,
            "sample_hash": self.sample_hash,
        }


@dataclass(frozen=True, slots=True)
class ImportedSample:
    sample: CanonicalSample
    trace: ImportTrace


class SampleImporter(Protocol):
    importer_id: str
    importer_version: str

    def import_record(
        self, record: RawRecord, config: Mapping[str, Any]
    ) -> CanonicalSample: ...


class CanonicalImporter:
    importer_id = "canonical"
    importer_version = "1.0.0"

    def import_record(
        self, record: RawRecord, config: Mapping[str, Any]
    ) -> CanonicalSample:
        if config:
            unknown = ", ".join(sorted(config))
            raise DataImportError(
                f"canonical importer does not accept config fields: {unknown}"
            )
        return parse_sample(record.value)


class ImporterRegistry:
    def __init__(self) -> None:
        self._importers: dict[str, SampleImporter] = {}
        self.register(CanonicalImporter())

    def register(self, importer: SampleImporter) -> None:
        importer_id = getattr(importer, "importer_id", "")
        importer_version = getattr(importer, "importer_version", "")
        if not importer_id or not importer_version:
            raise DataImportError("importer must define importer_id/importer_version")
        if importer_id in self._importers:
            raise DataImportError(f"importer {importer_id!r} is already registered")
        if not callable(getattr(importer, "import_record", None)):
            raise DataImportError("importer must implement import_record()")
        self._importers[importer_id] = importer

    def get(self, importer_id: str) -> SampleImporter:
        try:
            return self._importers[importer_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._importers))
            raise DataImportError(
                f"unknown importer {importer_id!r}; available: {available}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._importers))


def import_with_trace(
    *,
    importer: SampleImporter,
    record: RawRecord,
    config: Mapping[str, Any],
    dataset_id: str,
    source_fingerprint: str,
) -> ImportedSample:
    sample = importer.import_record(record, config)
    return ImportedSample(
        sample=sample,
        trace=ImportTrace(
            dataset_id=dataset_id,
            source_uri=record.source_uri,
            source_fingerprint=source_fingerprint,
            record_index=record.record_index,
            byte_offset=record.byte_offset,
            importer_id=importer.importer_id,
            importer_version=importer.importer_version,
            sample_id=sample.id,
            sample_hash=canonical_hash(sample),
        ),
    )
