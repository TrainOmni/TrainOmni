from __future__ import annotations

import pytest

from trainomni.modules.data.adapters.msswift.config import MSSwiftAdapterConfig
from trainomni.modules.data.collation.multimodal.config import (
    MultimodalCollatorConfig,
)
from trainomni.modules.data.model_io.transformers.config import (
    TransformersModelIOConfig,
)
from trainomni.modules.data.packing.sequence.config import SequencePackerConfig
from trainomni.modules.data.sources.arrow.config import ArrowSourceConfig
from trainomni.modules.data.sources.jsonl.config import JsonlSourceConfig
from trainomni.modules.data.sources.memory.config import MemorySourceConfig
from trainomni.modules.data.sources.parquet.config import ParquetSourceConfig
from trainomni.modules.data.supervision.causal_lm.config import (
    CausalSupervisionConfig,
)
from trainomni.modules.data.transforms.image.config import ImageTransformConfig
from trainomni.modules.data.transforms.media.config import MediaTransformConfig
from trainomni.modules.data.transforms.video.config import VideoTransformConfig

SAMPLE = ({"sample_id": "s", "content": ({"kind": "text", "value": "x"},)},)


@pytest.mark.parametrize("config_type", [ParquetSourceConfig, ArrowSourceConfig])
def test_columnar_config_rejects_coercible_invalid_types(config_type) -> None:
    with pytest.raises(TypeError, match="columns"):
        config_type(dataset_id="d", paths=("x.parquet",), columns=(1,))
    with pytest.raises(TypeError, match="repeat"):
        config_type(dataset_id="d", paths=("x.parquet",), repeat="false")
    with pytest.raises(TypeError, match="batch_rows"):
        config_type(dataset_id="d", paths=("x.parquet",), batch_rows=True)


def test_source_configs_reject_non_boolean_repeat() -> None:
    with pytest.raises(TypeError, match="repeat"):
        MemorySourceConfig(samples=SAMPLE, repeat="false")
    with pytest.raises(TypeError, match="repeat"):
        JsonlSourceConfig(path="x.jsonl", sha256="a" * 64, repeat="false")


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: MediaTransformConfig(require_sha256=1), "require_sha256"),
        (lambda: MSSwiftAdapterConfig(decode_image_bytes="true"), "decode_image_bytes"),
        (
            lambda: TransformersModelIOConfig(
                processor_name_or_path="p", trust_remote_code=1
            ),
            "trust_remote_code",
        ),
        (
            lambda: TransformersModelIOConfig(
                processor_name_or_path="p", local_files_only="false"
            ),
            "local_files_only",
        ),
        (
            lambda: TransformersModelIOConfig(
                processor_name_or_path="p", add_generation_prompt=1
            ),
            "add_generation_prompt",
        ),
        (
            lambda: TransformersModelIOConfig(
                processor_name_or_path="p", require_assistant_mask=0
            ),
            "require_assistant_mask",
        ),
    ],
)
def test_data_module_flags_require_actual_booleans(factory, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        factory()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ImageTransformConfig(max_pixels=True), "max_pixels"),
        (lambda: VideoTransformConfig(frames=True), "frames"),
        (
            lambda: VideoTransformConfig(frames=1, max_decoded_frames=True),
            "max_decoded_frames",
        ),
        (lambda: SequencePackerConfig(max_length=True, pad_token_id=0), "max_length"),
        (lambda: SequencePackerConfig(max_length=8, pad_token_id=True), "pad_token_id"),
        (
            lambda: SequencePackerConfig(
                max_length=8, pad_token_id=0, max_samples_per_pack=True
            ),
            "max_samples_per_pack",
        ),
        (lambda: MultimodalCollatorConfig(pad_to_multiple_of=True), "pad_to_multiple_of"),
        (lambda: CausalSupervisionConfig(ignore_index=True), "ignore_index"),
    ],
)
def test_data_module_integer_fields_reject_booleans(factory, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        factory()


def test_sequence_fields_are_canonicalized_and_strict() -> None:
    config = TransformersModelIOConfig(
        processor_name_or_path="p",
        assistant_mask_fields=["assistant_mask"],
    )
    assert config.assistant_mask_fields == ("assistant_mask",)
    with pytest.raises(TypeError, match="assistant_mask_fields"):
        TransformersModelIOConfig(
            processor_name_or_path="p",
            assistant_mask_fields=[1],
        )
    with pytest.raises(TypeError, match="metadata_columns"):
        MSSwiftAdapterConfig(metadata_columns=[1])
    with pytest.raises(ValueError, match=r"reserved trainomni\.\*"):
        MSSwiftAdapterConfig(
            metadata_columns=["trainomni.dataset", "trainomni.position"]
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SequencePackerConfig(
            max_length=8,
            pad_token_id=0,
            sequence_fields=("token_type_ids",),
            field_pad_values={"token_type_ids": float("nan")},
        ),
        lambda: MultimodalCollatorConfig(
            field_pad_values={"float_features": float("nan")}
        ),
    ],
)
def test_padding_configuration_rejects_nan(factory) -> None:
    with pytest.raises(ValueError, match="must not be NaN"):
        factory()
