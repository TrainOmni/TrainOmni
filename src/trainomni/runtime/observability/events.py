"""Append-only structured event sink."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlEventWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: Mapping[str, Any]) -> None:
        record = {
            "event": event, **payload,
            "timestamp": datetime.now(UTC).isoformat(),
            "monotonic_seconds": time.perf_counter(),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


class NullEventWriter:
    """Non-primary rank sink with the same narrow event interface."""

    def write(self, event: str, payload: Mapping[str, Any]) -> None:
        del event, payload
