"""Modal encoder protocol."""

from typing import Any, Protocol

from trainomni.contracts.features import ModalFeatures


class EncoderModule(Protocol):
    def __call__(self, modal_inputs: Any) -> ModalFeatures: ...
