"""Strict YAML/JSON task and run loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from trainomni.core.errors import SpecError

from .run import RunSpec
from .task import TaskSpec

SpecT = TypeVar("SpecT", TaskSpec, RunSpec)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SpecError(f"spec file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError(f"invalid JSON spec {path}: {exc}") from exc
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SpecError("YAML loading requires PyYAML") from exc
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SpecError(f"invalid YAML spec {path}: {exc}") from exc
    else:
        raise SpecError(f"unsupported spec extension: {path.suffix}")
    if not isinstance(value, Mapping):
        raise SpecError(f"spec root must be a mapping: {path}")
    return value


def load_task(path: str | Path) -> TaskSpec:
    return TaskSpec.from_mapping(_load_mapping(Path(path)))


def load_run(path: str | Path) -> RunSpec:
    return RunSpec.from_mapping(_load_mapping(Path(path)))
