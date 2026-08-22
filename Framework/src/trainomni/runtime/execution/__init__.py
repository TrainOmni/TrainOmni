"""Execution backends and distributed process lifecycle."""

from .factory import build_execution_backend
from .process import ProcessContext
from .protocol import ExecutionBackend

__all__ = ["ExecutionBackend", "ProcessContext", "build_execution_backend"]
