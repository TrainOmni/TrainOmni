"""In-memory artifact catalog used by orchestration and tests."""

from __future__ import annotations

from trainomni.contracts import ArtifactManifest, ArtifactRef


class ArtifactCatalog:
    def __init__(self) -> None:
        self._items: dict[str, ArtifactManifest] = {}

    def register(self, manifest: ArtifactManifest) -> None:
        existing = self._items.get(manifest.artifact_id)
        if existing is not None and existing != manifest:
            raise ValueError(f"artifact {manifest.artifact_id!r} is already registered")
        for parent in manifest.parents:
            if parent.artifact_id not in self._items:
                raise ValueError(
                    f"artifact parent {parent.artifact_id!r} is not registered"
                )
        self._items[manifest.artifact_id] = manifest

    def resolve(self, reference: ArtifactRef) -> ArtifactManifest:
        try:
            return self._items[reference.artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact {reference.artifact_id!r}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
