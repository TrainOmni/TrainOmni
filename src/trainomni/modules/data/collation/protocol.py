"""Final batch collation boundary."""

from collections.abc import Sequence
from typing import Protocol

from trainomni.contracts.batch import OmniBatch, SupervisedExample


class Collator(Protocol):
    def collate(self, examples: Sequence[SupervisedExample]) -> OmniBatch: ...
