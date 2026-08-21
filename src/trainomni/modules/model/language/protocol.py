"""Language component protocol."""

from typing import Any, Protocol


class LanguageModule(Protocol):
    def embed(self, input_ids: Any) -> Any: ...

    def forward_embeddings(
        self, embeddings: Any, *, attention_mask: Any | None = None, **kwargs: Any
    ) -> Any: ...
