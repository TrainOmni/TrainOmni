"""Immutable external-asset provenance used by builtin Transformers modules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from trainomni.specs.digest import identity_digest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def validate_asset_fields(
    *, revision: str | None, asset_manifest_sha256: str | None
) -> None:
    if revision is not None and not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError(
            "revision must be an immutable 40-64 character lowercase commit digest"
        )
    if asset_manifest_sha256 is not None and not _SHA256.fullmatch(
        asset_manifest_sha256
    ):
        raise ValueError("asset_manifest_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class AssetProvenance:
    reproducible: bool
    lock_entries: dict[str, str]
    issues: tuple[str, ...]


_TRANSFORMERS_ASSET_MODULES = frozenset(
    {
        "model:trainomni/monolithic_transformers@1",
        "encoder:trainomni/transformers_vision@1",
        "encoder:trainomni/transformers_video@1",
        "language:trainomni/transformers_causal_lm@1",
        "model_io:trainomni/transformers@1",
    }
)
_COLUMNAR_DATA_MODULES = frozenset(
    {
        "data_source:trainomni/parquet@1",
        "data_source:trainomni/arrow@1",
    }
)


def _resolved_location(value: object, *, task_root: Path | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute() and task_root is not None:
        path = task_root / path
    return str(path.resolve())


def _transformers_asset_location(
    config, *, task_root: Path | None
) -> tuple[object, bool]:
    location = config.get(
        "model_name_or_path",
        config.get("processor_name_or_path"),
    )
    local_only = config.get("local_files_only") is True
    if not isinstance(location, str) or not location:
        return location, local_only
    path = Path(location)
    candidate = path if path.is_absolute() or task_root is None else task_root / path
    is_local = local_only or candidate.is_dir()
    if is_local:
        return _resolved_location(location, task_root=task_root), True
    return location, False


def _columnar_path_identity(config, *, task_root: Path | None) -> object:
    paths = config.get("paths")
    if not isinstance(paths, (tuple, list)):
        return paths
    if task_root is None:
        return tuple(paths)
    return tuple(
        _resolved_location(path, task_root=task_root)
        if isinstance(path, str)
        else path
        for path in paths
    )


def task_asset_provenance(task, *, task_root: Path | None = None) -> AssetProvenance:
    entries = {}
    issues = []
    for index, reference in enumerate(task.module_refs()):
        module_id = str(reference.module_id)
        if module_id in _COLUMNAR_DATA_MODULES:
            manifest = reference.config.get("dataset_manifest_sha256")
            strict = bool(isinstance(manifest, str) and _SHA256.fullmatch(manifest))
            key = f"asset:{index:04d}:{module_id}"
            entries[key] = identity_digest(
                {
                    "module": module_id,
                    "dataset_id": reference.config.get("dataset_id"),
                    "paths": (
                        "<physical-columnar-paths>"
                        if strict
                        else _columnar_path_identity(
                            reference.config,
                            task_root=task_root,
                        )
                    ),
                    "dataset_manifest_sha256": manifest,
                    "reproducible": strict,
                }
            )
            if not strict:
                issues.append(f"{module_id} requires dataset_manifest_sha256")
            continue
        if module_id not in _TRANSFORMERS_ASSET_MODULES:
            continue
        revision = reference.config.get("revision")
        manifest = reference.config.get("asset_manifest_sha256")
        manifest_pinned = isinstance(manifest, str) and bool(
            _SHA256.fullmatch(manifest)
        )
        location, is_local = _transformers_asset_location(
            reference.config,
            task_root=task_root,
        )
        revision_pinned = isinstance(revision, str) and bool(
            _IMMUTABLE_REVISION.fullmatch(revision)
        )
        strict = bool(
            manifest_pinned
            or revision_pinned
            and not is_local
        )
        key = f"asset:{index:04d}:{module_id}"
        entries[key] = identity_digest(
            {
                "module": module_id,
                "location": (
                    "<physical-transformers-asset>"
                    if manifest_pinned
                    else location
                ),
                "revision": revision,
                "asset_manifest_sha256": manifest,
                "asset_kind": (
                    "manifest-pinned"
                    if manifest_pinned
                    else "local"
                    if is_local
                    else "remote"
                ),
                "reproducible": strict,
            }
        )
        if not strict:
            if is_local:
                issues.append(
                    f"{module_id} local assets require asset_manifest_sha256; "
                    "revision does not identify local payload bytes"
                )
            else:
                issues.append(
                    f"{module_id} requires an immutable revision or "
                    "asset_manifest_sha256"
                )
    return AssetProvenance(not issues, entries, tuple(issues))
