"""Parquet source configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class ParquetSourceConfig:
    dataset_id: str
    paths: tuple[str, ...]
    columns: tuple[str, ...] = ()
    batch_rows: int = 256
    repeat: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("Parquet dataset_id must not be empty")
        if not isinstance(self.paths, (tuple, list)) or not self.paths:
            raise ValueError("Parquet paths must be a non-empty sequence")
        if any(not isinstance(path, str) or not path for path in self.paths):
            raise ValueError("Parquet paths must contain non-empty strings")
        if not isinstance(self.columns, (tuple, list)):
            raise TypeError("Parquet columns must be a sequence")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Parquet columns must be unique")
        if self.batch_rows <= 0:
            raise ValueError("Parquet batch_rows must be positive")
        object.__setattr__(self, "dataset_id", self.dataset_id.strip())
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "columns", tuple(self.columns))
