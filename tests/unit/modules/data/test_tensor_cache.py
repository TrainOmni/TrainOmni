import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError, SpecError
from trainomni.modules.data.model_io.transformers.config import (
    TransformersModelIOConfig,
)
from trainomni.modules.data.model_io.transformers.module import TransformersModelIO
from trainomni.modules.data.transforms.tensor_cache.config import TensorCacheConfig
from trainomni.modules.data.transforms.tensor_cache.module import TensorCacheTransform
from trainomni.modules.objectives._ops.cache_identity import value_digest
from trainomni.modules.objectives.dense_kd.config import DenseKDConfig
from trainomni.modules.objectives.dense_kd.module import DenseKDObjective
from trainomni.modules.objectives.dpo.config import DPOConfig
from trainomni.modules.objectives.dpo.module import DPOObjective
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.loop.step import execute_forward_plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Processor:
    def __call__(self, **kwargs):
        del kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def make_cache(
    root: Path,
    *,
    tensor_digest: str | None = None,
    schema_version: int = 3,
):
    tensor_path = root / "cache.safetensors"
    save_file({"sample.teacher": torch.arange(21).reshape(3, 7)}, tensor_path)
    index = {
        "schema_version": schema_version,
        "samples": {
            "sample": {
                "file": "cache.safetensors",
                "sha256": tensor_digest or sha256(tensor_path),
                "tensors": {"teacher_logits": "sample.teacher"},
                "bindings": {
                    "teacher_logits": {
                        "input_ids_sha256": value_digest(torch.tensor([1, 2, 3])),
                        "attention_mask_sha256": value_digest(
                            torch.tensor([1, 1, 1])
                        ),
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


def test_tensor_cache_schema_v2_fails_closed(tmp_path: Path) -> None:
    index = make_cache(tmp_path, schema_version=2)
    with pytest.raises(SpecError, match="unsupported tensor-cache index schema"):
        TensorCacheTransform(
            TensorCacheConfig(
                index_path="index.json",
                index_sha256=sha256(index),
            ),
            task_root=tmp_path,
        )


def _batched_cache(transform: TensorCacheTransform) -> dict[str, torch.Tensor]:
    sample = transform.apply(OmniSample("sample", (ContentBlock("text", "x"),)))
    return {
        name: value.unsqueeze(0)
        for name, value in sample.metadata["tensor_cache"].items()
    }


def test_schema_v3_kd_cache_rejects_padding_relayout_before_forward(
    tmp_path: Path,
) -> None:
    producer_ids = torch.tensor([1, 2, 3, 0])
    producer_attention = torch.tensor([1, 1, 1, 0])
    producer_positions = torch.tensor([1, 2])
    tensor_path = tmp_path / "cache.safetensors"
    save_file({"sample.teacher": torch.zeros(3, 7)}, tensor_path)
    index = {
        "schema_version": 3,
        "samples": {
            "sample": {
                "file": tensor_path.name,
                "sha256": sha256(tensor_path),
                "tensors": {"teacher_logits": "sample.teacher"},
                "bindings": {
                    "teacher_logits": {
                        "input_ids_sha256": value_digest(producer_ids),
                        "attention_mask_sha256": value_digest(producer_attention),
                        "supervised_positions_sha256": value_digest(
                            producer_positions
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
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    transform = TensorCacheTransform(
        TensorCacheConfig(
            index_path=index_path.name,
            index_sha256=sha256(index_path),
        ),
        task_root=tmp_path,
    )
    consumer_ids = torch.tensor([[0, 1, 2, 3]])
    batch = OmniBatch(
        sample_ids=("sample",),
        model_inputs={
            "input_ids": consumer_ids,
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        },
        labels=torch.tensor([[-100, -100, 2, 3]]),
        supervision=_batched_cache(transform),
    )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 7)}

    policy = Policy()
    with pytest.raises(ObjectiveError, match="identity mismatch"):
        execute_forward_plan(
            model=policy,
            objective=DenseKDObjective(
                DenseKDConfig(producer_identity_sha256="a" * 64)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0


def test_schema_v3_dpo_cache_rejects_padding_relayout_before_forward(
    tmp_path: Path,
) -> None:
    producer_attention = torch.tensor([1, 1, 1, 0])
    producer_positions = torch.tensor([1, 2])
    producer_chosen = torch.tensor([1, 2, 3, 0])
    producer_rejected = torch.tensor([1, 2, 4, 0])
    tensor_path = tmp_path / "cache.safetensors"
    save_file(
        {
            "sample.chosen": torch.zeros(3),
            "sample.rejected": torch.zeros(3),
        },
        tensor_path,
    )

    def binding(input_ids, targets, branch):
        return {
            "input_ids_sha256": value_digest(input_ids),
            "attention_mask_sha256": value_digest(producer_attention),
            "supervised_positions_sha256": value_digest(producer_positions),
            "target_token_ids_sha256": value_digest(targets),
            "producer_identity_sha256": "b" * 64,
            "branch": branch,
        }

    index = {
        "schema_version": 3,
        "samples": {
            "sample": {
                "file": tensor_path.name,
                "sha256": sha256(tensor_path),
                "tensors": {
                    "chosen_reference_logps": "sample.chosen",
                    "rejected_reference_logps": "sample.rejected",
                },
                "bindings": {
                    "chosen_reference_logps": binding(
                        producer_chosen,
                        torch.tensor([2, 3]),
                        "chosen",
                    ),
                    "rejected_reference_logps": binding(
                        producer_rejected,
                        torch.tensor([2, 4]),
                        "rejected",
                    ),
                },
            }
        },
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    transform = TensorCacheTransform(
        TensorCacheConfig(
            index_path=index_path.name,
            index_sha256=sha256(index_path),
        ),
        task_root=tmp_path,
    )
    supervision = _batched_cache(transform)
    consumer_attention = torch.tensor([[0, 1, 1, 1]])
    chosen = torch.tensor([[0, 1, 2, 3]])
    rejected = torch.tensor([[0, 1, 2, 4]])
    supervision.update(
        {
            "chosen_inputs": {
                "input_ids": chosen,
                "attention_mask": consumer_attention,
            },
            "rejected_inputs": {
                "input_ids": rejected,
                "attention_mask": consumer_attention,
            },
            "chosen_labels": torch.tensor([[-100, -100, 2, 3]]),
            "rejected_labels": torch.tensor([[-100, -100, 2, 4]]),
        }
    )
    batch = OmniBatch(
        sample_ids=("sample",),
        model_inputs={"input_ids": chosen},
        labels=chosen,
        supervision=supervision,
    )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 7)}

    policy = Policy()
    with pytest.raises(ObjectiveError, match="identity mismatch"):
        execute_forward_plan(
            model=policy,
            objective=DPOObjective(
                DPOConfig(reference_producer_identity_sha256="b" * 64)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0
