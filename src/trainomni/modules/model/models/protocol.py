"""Executable model protocol."""

from typing import Any, Protocol


class ModelModule(Protocol):
    def __call__(self, **model_inputs: Any) -> Any: ...
