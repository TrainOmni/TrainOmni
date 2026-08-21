"""Artifact identity records."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    kind: str
    uri: str
    digest: str

    def __post_init__(self) -> None:
        if not self.kind or not self.uri or not self.digest:
            raise ValueError("artifact kind, uri, and digest must not be empty")
