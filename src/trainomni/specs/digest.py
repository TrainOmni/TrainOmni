"""Canonical identity helpers for task, run, modules, and artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


def canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(inner)
            for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [canonical_value(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted(canonical_value(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def identity_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
