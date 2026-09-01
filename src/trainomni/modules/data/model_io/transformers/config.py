"""Basic Transformers processor configuration."""

from dataclasses import dataclass

from trainomni.core.assets import validate_asset_fields


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

    def __post_init__(self) -> None:
        if not self.processor_name_or_path:
            raise ValueError("processor_name_or_path must not be empty")
        validate_asset_fields(
            revision=self.revision,
            asset_manifest_sha256=self.asset_manifest_sha256,
        )
        if self.conversation_mode not in {"auto", "required", "disabled"}:
            raise ValueError("conversation_mode must be auto, required, or disabled")
        if not self.assistant_mask_fields or any(
            not field for field in self.assistant_mask_fields
        ):
            raise ValueError("assistant_mask_fields must contain non-empty names")
        if not self.loss_mask_field:
            raise ValueError("loss_mask_field must not be empty")
        if self.supervision_metadata_key is not None and not self.supervision_metadata_key:
            raise ValueError("supervision_metadata_key must be null or non-empty")
