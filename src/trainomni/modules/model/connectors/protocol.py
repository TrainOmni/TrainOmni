"""Modal connector protocol."""

from typing import Protocol

from trainomni.contracts.features import ModalFeatures


class ConnectorModule(Protocol):
    def __call__(self, features: ModalFeatures) -> ModalFeatures: ...
