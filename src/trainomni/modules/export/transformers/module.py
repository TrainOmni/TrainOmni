"""Atomic Transformers-native exporter for monolithic model adapters."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from trainomni.artifacts.lineage import directory_tree_sha256
from trainomni.contracts.artifact import ArtifactIdentity
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TransformersExportConfig


class TransformersExporter:
    def __init__(self, config: TransformersExportConfig) -> None:
        self.config = config

    @staticmethod
    def _unwrap(model):
        candidate = getattr(model, "model", None)
        if candidate is None:
            raise SpecError(
                "Transformers export requires a monolithic adapter exposing .model"
            )
        save_pretrained = getattr(candidate, "save_pretrained", None)
        if not callable(save_pretrained):
            raise SpecError("wrapped model does not implement save_pretrained")
        return candidate, save_pretrained

    def export(self, *, model, destination: Path, identity, processor=None):
        candidate, save_pretrained = self._unwrap(model)
        if self.config.save_processor and processor is None:
            raise SpecError(
                "Transformers export was configured to save the processor, but the "
                "task ModelIO exposes no processor"
            )
        destination = Path(destination)
        if destination.exists():
            raise SpecError(f"refusing to overwrite export: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        try:
            save_pretrained(
                staging,
                safe_serialization=self.config.safe_serialization,
                max_shard_size=self.config.max_shard_size,
            )
            if self.config.save_processor:
                save_processor = getattr(processor, "save_pretrained", None)
                if not callable(save_processor):
                    raise SpecError("task processor does not implement save_pretrained")
                save_processor(staging)
            payload_digest = directory_tree_sha256(staging)
            manifest = {
                "schema_version": 1,
                "kind": "transformers",
                "payload_tree_sha256": payload_digest,
                "identity": dict(sorted(identity.items())),
                "model_class": (
                    f"{candidate.__class__.__module__}.{candidate.__class__.__qualname__}"
                ),
                "safe_serialization": self.config.safe_serialization,
                "max_shard_size": self.config.max_shard_size,
                "processor_saved": self.config.save_processor,
            }
            temporary = staging / ".trainomni-export.json.tmp"
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, staging / "trainomni-export.json")
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ArtifactIdentity(
            kind="transformers",
            uri=str(destination.resolve()),
            digest=payload_digest,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("exporter:trainomni/transformers@1"),
        config_type=TransformersExportConfig,
        factory=lambda config, context: TransformersExporter(config),
        provides=CapabilitySet.of({"export.transformers"}),
        requires=CapabilitySet.of({"model.monolithic"}),
    )
