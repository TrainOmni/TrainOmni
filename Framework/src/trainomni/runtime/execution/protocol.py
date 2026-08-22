"""Task-agnostic execution backend contract."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class ExecutionBackend(Protocol):
    name: str
    canonical_model: Any
    execution_model: Any
    optimizer: Any
    scheduler: Any | None
    process: Any

    def accumulation_context(self, *, final_microbatch: bool) -> AbstractContextManager: ...

    def backward(self, loss: Any, scaler: Any | None) -> None: ...

    def unscale_gradients(self, scaler: Any | None) -> None: ...

    def clip_grad_norm(self, max_norm: float | None) -> float: ...

    def step(self, scaler: Any | None) -> None: ...

    def zero_grad(self) -> None: ...

    def metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
