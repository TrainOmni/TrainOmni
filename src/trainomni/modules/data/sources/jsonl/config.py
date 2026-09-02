"""JSONL canonical source configuration."""

from dataclasses import dataclass

from trainomni.modules.data._validation import require_bool, require_string


@dataclass(frozen=True, slots=True, kw_only=True)
class JsonlSourceConfig:
    path: str
    sha256: str
    repeat: bool = True

    def __post_init__(self) -> None:
        require_string(self.path, field="JSONL path")
        require_string(self.sha256, field="JSONL sha256")
        require_bool(self.repeat, field="JSONL repeat")
        if (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("JSONL sha256 must be a lowercase SHA-256 digest")
