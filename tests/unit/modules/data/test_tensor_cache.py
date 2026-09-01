import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.errors import SpecError
from trainomni.modules.data.model_io.transformers.config import (
    TransformersModelIOConfig,
)
from trainomni.modules.data.model_io.transformers.module import TransformersModelIO
from trainomni.modules.data.transforms.tensor_cache.config import TensorCacheConfig
from trainomni.modules.data.transforms.tensor_cache.module import TensorCacheTransform
from trainomni.modules.objectives._ops.cache_identity import value_digest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Processor:
    def __call__(self, **kwargs):
        del kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def make_cache(root: Path, *, tensor_digest: str | None = None):
    tensor_path = root / "cache.safetensors"
    save_file({"sample.teacher": torch.arange(21).reshape(3, 7)}, tensor_path)
    index = {
        "schema_version": 2,
        "samples": {
            "sample": {
                "file": "cache.safetensors",
                "sha256": tensor_digest or sha256(tensor_path),
                "tensors": {"teacher_logits": "sample.teacher"},
                "bindings": {
                    "teacher_logits": {
                        "input_ids_sha256": value_digest(torch.tensor([1, 2, 3])),
                        "supervised_positions_sha256": value_digest(
                            torch.tensor([1, 2])
                        ),
                        "target_token_ids_sha256": value_digest(
                            torch.tensor([2, 3])
                        ),
                        "producer_identity_sha256": "a" * 64,
                        "branch": "teacher",
                    }
                },
            }
        },
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    return index_path


def test_hash_pinned_tensor_cache_reaches_model_io_supervision(tmp_path: Path) -> None:
    index = make_cache(tmp_path)
    transform = TensorCacheTransform(
        TensorCacheConfig(
            index_path="index.json",
            index_sha256=sha256(index),
        ),
        task_root=tmp_path,
    )
    sample = transform.apply(
        OmniSample("sample", (ContentBlock("text", "distill"),))
    )
    encoded = TransformersModelIO(
        Processor(),
        TransformersModelIOConfig(
            processor_name_or_path="unused",
            supervision_metadata_key="tensor_cache",
        ),
    ).encode(sample)
    assert encoded.supervision["teacher_logits"].shape == (3, 7)
    assert torch.equal(
        encoded.supervision["teacher_logits"], torch.arange(21).reshape(3, 7)
    )
    assert encoded.supervision[
        "__cache_identity__teacher_logits__producer_identity_sha256"
    ].shape == (32,)


def test_tensor_cache_corruption_fails_before_model_io(tmp_path: Path) -> None:
    index = make_cache(tmp_path, tensor_digest="0" * 64)
    transform = TensorCacheTransform(
        TensorCacheConfig(
            index_path="index.json",
            index_sha256=sha256(index),
        ),
        task_root=tmp_path,
    )
    with pytest.raises(SpecError, match="file digest mismatch"):
        transform.apply(OmniSample("sample", (ContentBlock("text", "x"),)))
