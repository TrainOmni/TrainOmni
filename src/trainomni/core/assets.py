"""Immutable external-asset provenance used by builtin Transformers modules."""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def task_asset_provenance(task) -> AssetProvenance:
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
        strict = bool(
            manifest_pinned
            or isinstance(revision, str)
            and _IMMUTABLE_REVISION.fullmatch(revision)
        )
        key = f"asset:{index:04d}:{module_id}"
        entries[key] = identity_digest(
                {
                    "module": module_id,
                    "location": (
                        "<physical-transformers-asset>"
                        if manifest_pinned
                        else reference.config.get(
                            "model_name_or_path",
                            reference.config.get("processor_name_or_path"),
                        )
                    ),
                "revision": revision,
                "asset_manifest_sha256": manifest,
                "reproducible": strict,
            }
        )
        if not strict:
            issues.append(
                f"{module_id} requires an immutable revision or asset_manifest_sha256"
            )
    return AssetProvenance(not issues, entries, tuple(issues))
