"""Model-plugin-owned export with a core-owned reproducibility manifest."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trainomni.checkpoint import (
    DCP_CHECKPOINT_VERSION,
    LOCAL_CHECKPOINT_VERSION,
    DCPCheckpointManager,
    LocalCheckpointManager,
    ObjectState,
    StateRegistry,
)
from trainomni.config import ResolvedRunSpec
from trainomni.models import ModelBuildContext, ModelBundle

from .seed import seed_everything


class ExportError(RuntimeError):
    pass


def export_model(
    resolved: ResolvedRunSpec,
    plugin: Any,
    *,
    checkpoint: Path,
    output_dir: Path,
    export_format: str,
    trusted_checkpoint: bool = False,
    config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if export_format not in resolved.plugin_manifest.capabilities.export_formats:
        raise ExportError(
            f"model plugin does not support export format {export_format!r}"
        )
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise ExportError(f"checkpoint does not exist: {checkpoint}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(resolved.run.seed, resolved.run.stage.engine.config)
    bundle = plugin.build(
        ModelBuildContext(
            config=resolved.run.model.config,
            stage_id=resolved.run.stage.stage_id,
            output_dir=output_dir,
            mode="export",
        )
    )
    if not isinstance(bundle, ModelBundle):
        raise ExportError("plugin.build() must return ModelBundle for export")
    load_checkpoint_weights(
        checkpoint, bundle, trusted_checkpoint=trusted_checkpoint
    )
    result = plugin.export(
        bundle,
        checkpoint,
        {"format": export_format, "output_dir": output_dir, **dict(config or {})},
    )
    payload = {
        "schema_version": "trainomni.export.v1",
        "run_fingerprint": resolved.fingerprint,
        "plugin": {
            "id": resolved.plugin_manifest.plugin_id,
            "version": resolved.plugin_manifest.plugin_version,
        },
        "checkpoint": str(checkpoint),
        "format": export_format,
        "output_dir": str(output_dir),
        "plugin_result": _json_safe(result),
    }
    target = output_dir / "export-manifest.json"
    temporary = target.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return payload


def load_checkpoint_weights(
    checkpoint: Path, bundle: ModelBundle, *, trusted_checkpoint: bool
) -> None:
    manifest_path = checkpoint / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"invalid checkpoint manifest: {checkpoint}") from exc
    schema = manifest.get("schema_version")
    if schema == DCP_CHECKPOINT_VERSION:
        try:
            import torch
        except ImportError as exc:
            raise ExportError("DCP model loading requires PyTorch") from exc
        DCPCheckpointManager(checkpoint.parent, torch).load_model(
            checkpoint.name, bundle.model
        )
        return
    if schema != LOCAL_CHECKPOINT_VERSION:
        raise ExportError(f"unsupported checkpoint schema {schema!r}")
    registry = StateRegistry()
    registry.register("model", ObjectState(bundle.model))
    LocalCheckpointManager(checkpoint.parent).load(
        checkpoint.name,
        registry,
        trusted=trusted_checkpoint,
        strict=False,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)[:500]}
