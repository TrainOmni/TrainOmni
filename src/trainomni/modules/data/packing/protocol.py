"""Packing boundary."""

from collections.abc import Mapping
from typing import Any, Protocol

from trainomni.contracts.batch import SupervisedExample


class Packer(Protocol):
    def add(self, sample: SupervisedExample) -> tuple[SupervisedExample, ...]: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
