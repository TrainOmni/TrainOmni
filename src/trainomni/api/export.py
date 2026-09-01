"""Stable checkpoint-to-artifact export operation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from trainomni.artifacts.lineage import file_sha256
from trainomni.contracts.artifact import ArtifactIdentity
from trainomni.core.errors import SpecError
from trainomni.runtime.random import seed_everything
from trainomni.specs.digest import identity_digest

from ._checkpoint import load_model_checkpoint
from .train import assemble, load_resolved_run


@dataclass(frozen=True, slots=True)
class ExportResult:
    exporter: str
    checkpoint: Path
    artifact: ArtifactIdentity


def export_artifact(
    *,
    task_path: str | Path,
    run_path: str | Path,
    checkpoint: str | Path,
    exporter: str | None = None,
    destination: str | Path | None = None,
    allow_local_code: bool = False,
) -> ExportResult:
    run = load_resolved_run(run_path)
    seed_everything(run.seed, deterministic=run.deterministic)
    task, assembly = assemble(
        task_path=task_path,
        allow_local_code=allow_local_code,
    )
    if not assembly.exporters:
        raise SpecError("task defines no exporters")
    available = dict(assembly.exporters)
    if exporter is None:
        if len(available) != 1:
            raise SpecError("exporter must be selected when a task defines multiple exporters")
        exporter_id, implementation = next(iter(available.items()))
    else:
        try:
            implementation = available[exporter]
        except KeyError as exc:
            raise SpecError(
                f"unknown task exporter {exporter!r}; available: {', '.join(available)}"
            ) from exc
        exporter_id = exporter
    model, _, _, checkpoint_path, checkpoint_manifest_object = load_model_checkpoint(
        task=task,
        assembly=assembly,
        run=run,
        checkpoint=checkpoint,
        restore_objective=False,
    )
    manifest_path = checkpoint_path / "manifest.json"
    checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if checkpoint_manifest != checkpoint_manifest_object.to_mapping():
        raise SpecError("checkpoint manifest changed during export")
    target = (
        Path(destination).resolve()
        if destination is not None
        else run.checkpoint.directory.parent
        / "exports"
        / checkpoint_path.name
        / exporter_id.split(":", 1)[-1].replace("/", "_").replace("@", "_")
    )
    identity = {
        "task_digest": task.digest,
        "run_digest": run.digest,
        "checkpoint_model_sha256": str(checkpoint_manifest["model_sha256"]),
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_global_step": str(checkpoint_manifest["global_step"]),
        "module_lock_digest": identity_digest(assembly.module_lock),
        "exporter": exporter_id,
    }
    artifact = implementation.export(
        model=model,
        destination=target,
        identity=identity,
        processor=getattr(assembly.stream.model_io, "processor", None),
    )
    return ExportResult(
        exporter=exporter_id,
        checkpoint=checkpoint_path,
        artifact=artifact,
    )
