"""Data-adapter protocol."""

from typing import Protocol

from trainomni.contracts.data import DataRecord
from trainomni.contracts.sample import OmniSample


class DataAdapter(Protocol):
    def adapt(self, record: DataRecord) -> OmniSample: ...
