"""Stateful stream and module protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .batch import OmniBatch


class StatefulBatchStream(Protocol):
    def next_batch(self, batch_size: int) -> OmniBatch: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class StatefulModule(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
