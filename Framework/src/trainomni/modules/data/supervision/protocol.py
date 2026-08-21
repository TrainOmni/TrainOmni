"""Supervision construction boundary."""

from typing import Protocol

from trainomni.contracts.batch import EncodedSample, SupervisedExample


class Supervision(Protocol):
    def annotate(self, sample: EncodedSample) -> SupervisedExample: ...
