"""Safe JSON/YAML loading for TrainOmni run specifications."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from trainomni.contracts import ValidationIssue, ValidationReport

from .schema import RunSpec


class ConfigLoadError(ValueError):
    pass


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"cannot read config {path}: {exc}") from exc
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(raw)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(raw)
        else:
            raise ConfigLoadError(
                f"unsupported config extension {path.suffix!r}; use .json/.yaml/.yml"
            )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigLoadError(f"invalid config syntax in {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfigLoadError(f"config root in {path} must be an object/mapping")
    return value


def load_run_spec(path: str | Path) -> RunSpec:
    source = Path(path).resolve()
    value = _read_mapping(source)
    try:
        return RunSpec.model_validate(value)
    except ValidationError as exc:
        raise ConfigLoadError(format_pydantic_error(source, exc)) from exc


def validation_report_from_error(
    source: str | Path, error: ValidationError
) -> ValidationReport:
    issues = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        issues.append(
            ValidationIssue(
                code=f"config.{item['type']}",
                message=item["msg"],
                path=location or None,
                source=str(Path(source).resolve()),
            )
        )
    return ValidationReport(tuple(issues))


def format_pydantic_error(source: Path, error: ValidationError) -> str:
    report = validation_report_from_error(source, error)
    lines = [f"invalid TrainOmni config {source}:"]
    for issue in report.issues:
        location = f" at {issue.path}" if issue.path else ""
        lines.append(f"- [{issue.code}]{location}: {issue.message}")
    return "\n".join(lines)
