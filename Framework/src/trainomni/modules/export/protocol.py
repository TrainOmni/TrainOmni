"""Deployment artifact exporter protocol."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from trainomni.contracts.artifact import ArtifactIdentity


class Exporter(Protocol):
    def export(
        self,
        *,
        model: Any,
        destination: Path,
        identity: Mapping[str, str],
        processor: Any | None = None,
    ) -> ArtifactIdentity: ...
