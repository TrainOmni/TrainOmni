"""JSONL canonical source configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class JsonlSourceConfig:
    path: str
    sha256: str
    repeat: bool = True

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("JSONL path must not be empty")
        if (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("JSONL sha256 must be a lowercase SHA-256 digest")
