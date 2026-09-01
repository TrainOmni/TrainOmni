"""Stateful canonical sample-source protocol."""

from collections.abc import Mapping
from typing import Any, Protocol

from trainomni.contracts.data import DataRecord
from trainomni.contracts.sample import OmniSample


class SampleSource(Protocol):
    def next_sample(self) -> OmniSample: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def metrics(self) -> Mapping[str, int | float]: ...


class RecordSource(Protocol):
    def next_record(self) -> DataRecord: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def metrics(self) -> Mapping[str, int | float]: ...
