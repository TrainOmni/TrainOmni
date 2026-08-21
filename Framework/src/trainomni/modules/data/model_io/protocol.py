"""Model-specific encoding boundary."""

from typing import Protocol

from trainomni.contracts.batch import EncodedSample
from trainomni.contracts.sample import OmniSample


class ModelIO(Protocol):
    def encode(self, sample: OmniSample) -> EncodedSample: ...
