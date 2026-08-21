"""Generic full-state safetensors exporter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from trainomni.contracts.artifact import ArtifactIdentity
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import SafetensorsExportConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SafetensorsExporter:
    def __init__(self, config: SafetensorsExportConfig) -> None:
        self.config = config

    def export(self, *, model, destination: Path, identity, processor=None):
        del processor
        try:
            from safetensors import safe_open
            from safetensors.torch import save_model
        except ImportError as exc:
            raise SpecError("safetensors export requires safetensors") from exc
        destination = Path(destination)
        if destination.exists():
            raise SpecError(f"refusing to overwrite export: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        try:
            output = staging / self.config.filename
            save_model(model, output, metadata=dict(identity))
            digest = _sha256(output)
            with safe_open(output, framework="pt", device="cpu") as stream:
                tensor_count = len(tuple(stream.keys()))
            manifest = {
                "schema_version": 1,
                "kind": "safetensors",
                "file": self.config.filename,
                "sha256": digest,
                "tensor_count": tensor_count,
                "state_key_count": len(model.state_dict()),
                "identity": dict(sorted(identity.items())),
            }
            temporary_manifest = staging / ".manifest.json.tmp"
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, staging / "manifest.json")
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ArtifactIdentity(
            kind="safetensors",
            uri=str((destination / self.config.filename).resolve()),
            digest=digest,
        )


def load_safetensors_artifact(model, artifact: str | Path) -> None:
    artifact = Path(artifact)
    manifest_path = artifact / "manifest.json"
    if not manifest_path.is_file():
        raise SpecError(f"safetensors artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read safetensors artifact manifest: {exc}") from exc
    filename = manifest.get("file")
    digest = manifest.get("sha256")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "safetensors"
        or not isinstance(filename, str)
        or not isinstance(digest, str)
    ):
        raise SpecError("safetensors artifact manifest is invalid")
    path = artifact / filename
    if not path.is_file() or _sha256(path) != digest:
        raise SpecError("safetensors artifact digest mismatch")
    try:
        from safetensors.torch import load_model
    except ImportError as exc:
        raise SpecError("safetensors artifact loading requires safetensors") from exc
    try:
        missing, unexpected = load_model(model, path, strict=True, device="cpu")
    except Exception as exc:
        raise SpecError(f"safetensors artifact is incompatible: {exc}") from exc
    if missing or unexpected:
        raise SpecError(
            f"safetensors artifact keys differ: missing={missing}, unexpected={unexpected}"
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("exporter:trainomni/safetensors@1"),
        config_type=SafetensorsExportConfig,
        factory=lambda config, context: SafetensorsExporter(config),
        provides=CapabilitySet.of({"export.safetensors"}),
        requires=CapabilitySet.of({"model.parameters"}),
    )
