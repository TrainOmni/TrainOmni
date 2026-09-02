from __future__ import annotations

import pytest
import torch

from trainomni.contracts.batch import EncodedSample, OmniBatch, SupervisedExample
from trainomni.contracts.data import DataRecord
from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.errors import SpecError
from trainomni.modules.data.adapters.msswift.config import MSSwiftAdapterConfig
from trainomni.modules.data.adapters.msswift.module import MSSwiftAdapter
from trainomni.modules.data.sources.jsonl.module import _parse_sample
from trainomni.modules.data.sources.memory.module import _sample_from_mapping

CONTENT = [{"kind": "text", "value": "x"}]


@pytest.mark.parametrize("sample_id", [None, 0, {}, [], "   "])
@pytest.mark.parametrize("parser", [_parse_sample, _sample_from_mapping])
def test_builtin_parsers_reject_non_string_or_blank_sample_ids(
    parser, sample_id
) -> None:
    value = {"sample_id": sample_id, "content": CONTENT}
    with pytest.raises(SpecError, match="sample_id"):
        if parser is _parse_sample:
            parser(value, line_number=1)
        else:
            parser(value, 0)


@pytest.mark.parametrize("invalid", ["", 0, {}, []])
@pytest.mark.parametrize("parser", [_parse_sample, _sample_from_mapping])
def test_builtin_parsers_use_field_presence_for_content_message_exclusivity(
    parser, invalid
) -> None:
    value = {
        "sample_id": "s",
        "content": invalid,
        "messages": [{"role": "user", "content": CONTENT}],
    }
    with pytest.raises(SpecError, match="exactly one"):
        if parser is _parse_sample:
            parser(value, line_number=1)
        else:
            parser(value, 0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OmniSample("   ", (ContentBlock("text", "x"),)),
        lambda: DataRecord("   ", {"x": 1}, "source"),
        lambda: DataRecord("sample", {"x": 1}, "   "),
        lambda: EncodedSample("   ", {"input_ids": torch.tensor([1])}),
        lambda: SupervisedExample(
            "   ", {"input_ids": torch.tensor([1])}, torch.tensor([1])
        ),
        lambda: OmniBatch(
            ("   ",), {"input_ids": torch.tensor([[1]])}, torch.tensor([[1]])
        ),
    ],
)
def test_process_bound_contracts_reject_blank_identities(factory) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        factory()


def test_process_bound_contracts_trim_and_freeze_identity_containers() -> None:
    sample = OmniSample(" sample ", [ContentBlock("text", "x")])
    encoded = EncodedSample(" sample ", {"input_ids": torch.tensor([1])})
    supervised = SupervisedExample(
        " sample ", {"input_ids": torch.tensor([1])}, torch.tensor([1])
    )
    batch = OmniBatch(
        [" sample "], {"input_ids": torch.tensor([[1]])}, torch.tensor([[1]])
    )
    assert sample.sample_id == encoded.sample_id == supervised.sample_id == "sample"
    assert batch.sample_ids == ("sample",)
    assert isinstance(sample.content, tuple)


def test_msswift_row_ids_are_string_only_and_never_falsey_coerced() -> None:
    adapter = MSSwiftAdapter(MSSwiftAdapterConfig())

    def record(row_id) -> DataRecord:
        return DataRecord(
            "physical-id",
            {"id": row_id, "text": "sample"},
            "dataset",
        )

    assert adapter.adapt(record(None)).sample_id == "physical-id"
    assert adapter.adapt(record(" row-id ")).sample_id == "row-id"
    with pytest.raises(SpecError, match="sample id column.*string"):
        adapter.adapt(record(0))
    with pytest.raises(SpecError, match="sample id column.*string"):
        adapter.adapt(record("   "))


def test_msswift_media_mapping_rejects_ambiguous_bytes_and_path() -> None:
    adapter = MSSwiftAdapter(MSSwiftAdapterConfig(decode_image_bytes=False))
    record = DataRecord(
        "physical-id",
        {
            "text": "<image> sample",
            "images": [{"bytes": b"payload", "path": "image.png"}],
        },
        "dataset",
    )
    with pytest.raises(SpecError, match="both 'bytes' and 'path'"):
        adapter.adapt(record)
