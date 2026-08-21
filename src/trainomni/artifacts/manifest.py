"""Materialize immutable resolved task/run/module identities."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trainomni import __version__
from trainomni.core.errors import CheckpointError
from trainomni.specs.digest import canonical_value


def _write_exact(path: Path, payload: Any) -> None:
    content = json.dumps(
        canonical_value(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise CheckpointError(f"resolved run identity already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def materialize_run_identity(
    *,
    output_root: Path,
    task: Any,
    run: Any,
    module_lock: Mapping[str, str],
    parameter_selection: Any,
) -> None:
    root = Path(output_root)
    _write_exact(root / "resolved" / "task.resolved.json", task)
    _write_exact(root / "resolved" / "run.resolved.json", run)
    _write_exact(root / "resolved" / "modules.lock.json", module_lock)
    _write_exact(
        root / "resolved" / "parameters.json",
        {
            "trainable_numel": parameter_selection.trainable_numel,
            "trainable_names": parameter_selection.trainable_names,
            "frozen_names": parameter_selection.frozen_names,
            "groups": [
                {
                    "name": group.name,
                    "parameter_count": len(group.parameters),
                    "numel": sum(
                        parameter.numel() for parameter in group.parameters
                    ),
                }
                for group in parameter_selection.groups
            ],
        },
    )
    _write_exact(
        root / "run-manifest.json",
        {
            "schema_version": 1,
            "framework_version": __version__,
            "task_digest": task.digest,
            "run_digest": run.digest,
            "module_lock": dict(module_lock),
        },
    )
