"""Dependency-free structured metric/event logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlRunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def __call__(self, event: str, state: Any, values: Mapping[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "step": int(getattr(getattr(state, "step", None), "value", 0)),
            "microstep": int(
                getattr(getattr(state, "microstep", None), "value", 0)
            ),
            "tokens": int(getattr(getattr(state, "tokens", None), "value", 0)),
            "values": {
                str(key): _json_value(value) for key, value in values.items()
            },
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return repr(value)[:500]
