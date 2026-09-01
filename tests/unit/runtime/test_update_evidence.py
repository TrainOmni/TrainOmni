from __future__ import annotations

import pytest
import torch
from torch import nn

from trainomni.core.errors import OptimizationError
from trainomni.runtime.optimization.evidence import (
    _sample_indices,
    capture_update_snapshot,
    finalize_update_evidence,
)


class TwoValueModel(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 1.0], dtype=dtype))


def optimizer(model: nn.Module):
    return torch.optim.SGD(
        [{"params": list(model.parameters()), "group_name": "connector"}],
        lr=0.1,
    )


def test_tensor_digest_proves_update_when_the_sampled_element_is_unchanged() -> None:
    model = TwoValueModel(dtype=torch.bfloat16)
    update = optimizer(model)
    model.weight.grad = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    snapshot = capture_update_snapshot(
        model,
        update,
        sample_elements_per_group=1,
    )
    update.step()
    evidence = finalize_update_evidence(
        snapshot,
        model,
        required_groups=("connector",),
    )["connector"]
    assert evidence["gradient_norm"] == pytest.approx(1.0)
    assert evidence["changed_tensor_count"] == 1
    assert evidence["changed_tensors"] == ["weight"]
    assert evidence["sampled_elements"] == 1
    assert evidence["changed_sampled_elements"] == 0
    assert evidence["before_sha256"] != evidence["after_sha256"]
    assert evidence["parameter_dtypes"] == ["torch.bfloat16"]


def test_required_group_with_no_actual_update_fails_closed() -> None:
    model = TwoValueModel()
    update = optimizer(model)
    snapshot = capture_update_snapshot(
        model,
        update,
        sample_elements_per_group=2,
    )
    with pytest.raises(OptimizationError, match="no actual parameter update"):
        finalize_update_evidence(
            snapshot,
            model,
            required_groups=("connector",),
        )

    with pytest.raises(OptimizationError, match="do not exist"):
        finalize_update_evidence(
            snapshot,
            model,
            required_groups=("language",),
        )


def test_large_tensor_sampling_uses_exact_in_bounds_integer_indices() -> None:
    numel = 200_540_160
    indices = _sample_indices(numel=numel, count=4096)

    assert indices.device.type == "cpu"
    assert indices.dtype == torch.int64
    assert indices.shape == (4096,)
    assert indices[0].item() == 0
    assert indices[-1].item() == numel - 1
    assert bool((indices[1:] > indices[:-1]).all())
    assert int(indices.min()) >= 0
    assert int(indices.max()) < numel
