"""ms-swift-compatible schema adapter configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class MSSwiftAdapterConfig:
    sample_id_column: str = "id"
    messages_column: str = "messages"
    text_pairs_column: str = "texts"
    query_column: str = "query"
    response_column: str = "response"
    text_column: str = "text"
    system_column: str = "system"
    history_column: str = "history"
    images_column: str = "images"
    videos_column: str = "videos"
    audios_column: str = "audios"
    metadata_columns: tuple[str, ...] = ()
    media_without_placeholders: str = "prepend"
    decode_image_bytes: bool = True

    def __post_init__(self) -> None:
        column_names = (
            self.sample_id_column,
            self.messages_column,
            self.text_pairs_column,
            self.query_column,
            self.response_column,
            self.text_column,
            self.system_column,
            self.history_column,
            self.images_column,
            self.videos_column,
            self.audios_column,
        )
        if any(not isinstance(item, str) or not item for item in column_names):
            raise TypeError("ms-swift adapter column names must be non-empty strings")
        if not isinstance(self.metadata_columns, (tuple, list)):
            raise TypeError("metadata_columns must be a sequence")
        if len(self.metadata_columns) != len(set(self.metadata_columns)):
            raise ValueError("metadata_columns must be unique")
        if self.media_without_placeholders not in {"prepend", "error"}:
            raise ValueError(
                "media_without_placeholders must be 'prepend' or 'error'"
            )
        object.__setattr__(self, "metadata_columns", tuple(self.metadata_columns))
