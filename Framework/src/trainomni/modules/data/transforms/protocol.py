"""Canonical sample transform protocol."""

from typing import Protocol

from trainomni.contracts.sample import OmniSample


class SampleTransform(Protocol):
    def apply(self, sample: OmniSample) -> OmniSample: ...
