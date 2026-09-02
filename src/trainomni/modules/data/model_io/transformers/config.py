"""Basic Transformers processor configuration."""

from dataclasses import dataclass

from trainomni.core.assets import validate_asset_fields
from trainomni.modules.data._validation import (
    normalize_string_sequence,
    require_bool,
    require_string,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformersModelIOConfig:
    processor_name_or_path: str
    revision: str | None = None
    asset_manifest_sha256: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    text_separator: str = "\n"
    conversation_mode: str = "auto"
    add_generation_prompt: bool = False
    require_assistant_mask: bool = True
    assistant_mask_fields: tuple[str, ...] = (
        "assistant_masks",
        "assistant_mask",
    )
    loss_mask_field: str = "loss_mask"
    supervision_metadata_key: str | None = None
    batch_axis_fields: tuple[str, ...] = (
        "input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids",
    )

    def __post_init__(self) -> None:
        require_string(
            self.processor_name_or_path,
            field="processor_name_or_path",
        )
        require_bool(self.trust_remote_code, field="trust_remote_code")
        require_bool(self.local_files_only, field="local_files_only")
        require_string(self.text_separator, field="text_separator", allow_empty=True)
        require_bool(self.add_generation_prompt, field="add_generation_prompt")
        require_bool(self.require_assistant_mask, field="require_assistant_mask")
        validate_asset_fields(
            revision=self.revision,
            asset_manifest_sha256=self.asset_manifest_sha256,
        )
        if self.conversation_mode not in {"auto", "required", "disabled"}:
            raise ValueError("conversation_mode must be auto, required, or disabled")
        assistant_mask_fields = normalize_string_sequence(
            self.assistant_mask_fields,
            field="assistant_mask_fields",
            allow_empty_sequence=False,
        )
        require_string(self.loss_mask_field, field="loss_mask_field")
        if self.supervision_metadata_key is not None:
            require_string(
                self.supervision_metadata_key,
                field="supervision_metadata_key",
            )
        object.__setattr__(self, "assistant_mask_fields", assistant_mask_fields)
        object.__setattr__(
            self,
            "batch_axis_fields",
            normalize_string_sequence(self.batch_axis_fields, field="batch_axis_fields"),
        )
