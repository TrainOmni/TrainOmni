"""Local media resolution configuration."""

from dataclasses import dataclass

from trainomni.modules.data._validation import require_bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaTransformConfig:
    require_sha256: bool = False

    def __post_init__(self) -> None:
        require_bool(self.require_sha256, field="require_sha256")
