"""Held-out evaluator protocol."""

from collections.abc import Mapping
from typing import Any, Protocol

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.loss import LossBundle


class Evaluator(Protocol):
    def reset(self) -> None: ...

    def update(self, batch: OmniBatch, loss: LossBundle) -> None: ...

    def compute(self) -> Mapping[str, Any]: ...
