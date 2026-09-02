"""Parquet source configuration."""

from dataclasses import dataclass

from trainomni.core.assets import validate_asset_fields
from trainomni.modules.data._validation import (
    normalize_string_sequence,
    require_bool,
    require_int,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParquetSourceConfig:
    dataset_id: str
    paths: tuple[str, ...]
    columns: tuple[str, ...] = ()
    batch_rows: int = 256
    repeat: bool = True
    dataset_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("Parquet dataset_id must not be empty")
        paths = normalize_string_sequence(
            self.paths,
            field="Parquet paths",
            allow_empty_sequence=False,
        )
        columns = normalize_string_sequence(self.columns, field="Parquet columns")
        require_int(self.batch_rows, field="Parquet batch_rows", minimum=1)
        require_bool(self.repeat, field="Parquet repeat")
        validate_asset_fields(
            revision=None,
            asset_manifest_sha256=self.dataset_manifest_sha256,
        )
        object.__setattr__(self, "dataset_id", self.dataset_id.strip())
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "columns", columns)
