from pathlib import Path

import pytest
import torch
from torch import nn

from trainomni.core.errors import CheckpointError
from trainomni.runtime.checkpoint.manager import CheckpointManager


class Stateful:
    def __init__(self, state=None):
        self.state = dict(state or {})

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


def manager(root: Path) -> CheckpointManager:
    return CheckpointManager(
        root=root,
        task_digest="a" * 64,
        run_digest="b" * 64,
        module_lock={"model": "c" * 64},
    )


def test_split_checkpoint_model_only_load_does_not_touch_optimizer(tmp_path: Path) -> None:
    source = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.01, foreach=False)
    source(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    checkpoint = manager(tmp_path).save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=optimizer,
        objective=Stateful({"objective": 1}),
        stream=Stateful({"cursor": 2}),
        runtime_metadata={"route": "test"},
    )
    expected = {name: value.detach().clone() for name, value in source.state_dict().items()}

    with (checkpoint / "optimizer.pt").open("ab") as stream:
        stream.write(b"corrupt optimizer only")

    target = nn.Linear(3, 2)
    manifest = manager(tmp_path).load_model_only(
        checkpoint,
        model=target,
        map_location="cpu",
    )
    assert manifest.global_step == 1
    assert all(torch.equal(target.state_dict()[name], value) for name, value in expected.items())

    with pytest.raises(CheckpointError, match="optimizer.pt"):
        manager(tmp_path).load(
            checkpoint,
            model=nn.Linear(3, 2),
            optimizer=torch.optim.AdamW(nn.Linear(3, 2).parameters()),
            objective=Stateful(),
            stream=Stateful(),
            map_location="cpu",
        )


def test_model_only_objective_restore_validates_runtime_identity(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = manager(tmp_path).save(
        global_step=0,
        micro_step=0,
        model=model,
        optimizer=optimizer,
        objective=Stateful({"temperature": 2}),
        stream=Stateful({"cursor": 0}),
    )
    objective = Stateful()
    manager(tmp_path).load_model_only(
        checkpoint,
        model=nn.Linear(2, 2),
        map_location="cpu",
        objective=objective,
    )
    assert objective.state == {"temperature": 2}
