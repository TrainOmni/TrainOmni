"""Backend-neutral artifact references and immutable run lineage metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

ARTIFACT_SCHEMA_VERSION = "trainomni.artifact.v1"
RESUME_LEVELS = frozenset({"exact", "stage_boundary", "weights_only", "transfer"})


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    selector: str = "last"
    uri: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be blank")
        if not self.selector.strip():
            raise ValueError("selector must not be blank")
        if self.uri is not None and not self.uri.strip():
            raise ValueError("artifact URI must be non-blank or None")

    def __str__(self) -> str:
        return f"artifact://{self.artifact_id}/{self.selector}"


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifact_id: str
    artifact_type: str
    run_id: str
    stage_id: str
    fingerprint: str
    resume_level: str
    parents: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("artifact_id", "artifact_type", "run_id", "stage_id", "fingerprint"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if self.resume_level not in RESUME_LEVELS:
            raise ValueError(
                f"unsupported resume_level {self.resume_level!r}; "
                f"expected one of {sorted(RESUME_LEVELS)}"
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
