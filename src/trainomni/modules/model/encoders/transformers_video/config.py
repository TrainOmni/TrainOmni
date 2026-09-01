"""Transformers video encoder configuration."""

from dataclasses import dataclass

from trainomni.core.assets import validate_asset_fields


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformersVideoConfig:
    model_name_or_path: str
    revision: str | None = None
    asset_manifest_sha256: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    output_field: str = "last_hidden_state"
    input_field: str = "pixel_values_videos"
    temporal_grid_field: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name_or_path or not self.output_field or not self.input_field:
            raise ValueError("model_name_or_path, output_field and input_field are required")
        validate_asset_fields(
            revision=self.revision,
            asset_manifest_sha256=self.asset_manifest_sha256,
        )
