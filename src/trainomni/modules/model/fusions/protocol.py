"""Cross-modal fusion protocol."""

from typing import Any, Protocol

from trainomni.contracts.features import ModalFeatureSet


class FusionModule(Protocol):
    def __call__(
        self,
        *,
        language: Any,
        input_ids: Any,
        modal_features: ModalFeatureSet,
        attention_mask: Any | None,
        modal_positions: Any | None,
        **kwargs: Any,
    ) -> Any: ...
