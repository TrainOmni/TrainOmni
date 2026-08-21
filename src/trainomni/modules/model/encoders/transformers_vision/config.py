"""Transformers vision encoder configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformersVisionConfig:
    model_name_or_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    output_field: str = "last_hidden_state"
    drop_cls_token: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path or not self.output_field:
            raise ValueError("model_name_or_path and output_field must not be empty")
