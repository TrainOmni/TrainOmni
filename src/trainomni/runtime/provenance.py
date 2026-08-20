"""Capture a lightweight reproducibility envelope for every execution."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from trainomni.config import ResolvedRunSpec

_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "datasets",
    "safetensors",
    "pydantic",
)

_SECRET_KEY_PARTS = ("token", "password", "secret", "api_key", "apikey")


def write_provenance(resolved: ResolvedRunSpec, output_dir: Path) -> Path:
    payload: dict[str, Any] = {
        "schema_version": "trainomni.provenance.v1",
        "run_fingerprint": resolved.fingerprint,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _version(name) for name in _PACKAGES},
        "distributed": {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
            if key in os.environ
        },
        "config": _redact(resolved.to_dict()),
    }
    target = output_dir / "provenance.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return target


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _redact(value: Any) -> Any:
    """Remove credential values before resolved configuration is persisted."""

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized == "environment" or any(
                part in normalized for part in _SECRET_KEY_PARTS
            ):
                result[key] = "<redacted>"
            else:
                result[key] = _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value
