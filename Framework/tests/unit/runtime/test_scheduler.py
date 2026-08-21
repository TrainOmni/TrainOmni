import pytest
import torch

from trainomni.runtime.optimization.scheduler import build_scheduler
from trainomni.specs.run import SchedulerSpec


@pytest.mark.parametrize("name", ["constant", "linear", "cosine"])
def test_scheduler_state_roundtrip_and_minimum_ratio(name: str) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(
        SchedulerSpec(name=name, warmup_steps=1, min_lr_ratio=0.2),
        optimizer,
        total_steps=5,
    )
    observed = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    if name == "constant":
        assert observed == [1.0] * 5
    else:
        assert observed[-1] == pytest.approx(0.2)
        assert all(0.2 <= value <= 1.0 for value in observed)

    restored_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(()))], lr=1.0
    )
    restored = build_scheduler(
        SchedulerSpec(name=name, warmup_steps=1, min_lr_ratio=0.2),
        restored_optimizer,
        total_steps=5,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.state_dict() == scheduler.state_dict()
