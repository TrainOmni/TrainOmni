import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from trainomni import __version__
from trainomni.core.errors import CheckpointError
from trainomni.runtime.checkpoint.manager import CheckpointManager


class Stateful:
    def __init__(self, state=None):
        self.state = dict(state or {})

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


def manager(
    root: Path,
    *,
    task_digest: str = "a" * 64,
    run_digest: str = "b" * 64,
    compatible_run_digests: tuple[str, ...] = (),
    module_lock: dict[str, str] | None = None,
    framework_version: str = "0.1.0",
) -> CheckpointManager:
    return CheckpointManager(
        root=root,
        task_digest=task_digest,
        run_digest=run_digest,
        module_lock=module_lock or {"model": "c" * 64},
        compatible_run_digests=compatible_run_digests,
        framework_version=framework_version,
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


def test_model_only_load_allows_a_different_run_but_full_resume_does_not(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source.parameters())
    checkpoint = manager(tmp_path).save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=source_optimizer,
        objective=Stateful(),
        stream=Stateful(),
    )
    evaluation_manager = manager(tmp_path, run_digest="d" * 64)
    target = nn.Linear(2, 2)
    manifest = evaluation_manager.load_model_only(
        checkpoint,
        model=target,
        map_location="cpu",
    )
    assert manifest.run_digest == "b" * 64
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in source.state_dict().items()
    )

    resume_model = nn.Linear(2, 2)
    resume_optimizer = torch.optim.AdamW(resume_model.parameters())
    with pytest.raises(CheckpointError, match="run_digest"):
        evaluation_manager.load(
            checkpoint,
            model=resume_model,
            optimizer=resume_optimizer,
            objective=Stateful(),
            stream=Stateful(),
            map_location="cpu",
        )


@pytest.mark.parametrize(
    ("manager_kwargs", "mismatch"),
    (
        ({"task_digest": "e" * 64}, "task_digest"),
        ({"module_lock": {"model": "e" * 64}}, "module_lock"),
        ({"framework_version": "9.9.9"}, "framework_version"),
    ),
)
def test_model_only_load_still_rejects_non_run_identity_mismatch(
    tmp_path: Path,
    manager_kwargs: dict,
    mismatch: str,
) -> None:
    source = nn.Linear(2, 2)
    checkpoint = manager(tmp_path).save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=torch.optim.AdamW(source.parameters()),
        objective=Stateful(),
        stream=Stateful(),
    )
    with pytest.raises(CheckpointError, match=mismatch):
        manager(tmp_path, **manager_kwargs).load_model_only(
            checkpoint,
            model=nn.Linear(2, 2),
            map_location="cpu",
        )


def test_model_only_load_still_validates_model_file_digest(tmp_path: Path) -> None:
    source = nn.Linear(2, 2)
    checkpoint = manager(tmp_path).save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=torch.optim.AdamW(source.parameters()),
        objective=Stateful(),
        stream=Stateful(),
    )
    with (checkpoint / "model.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(CheckpointError, match="model.safetensors"):
        manager(tmp_path, run_digest="d" * 64).load_model_only(
            checkpoint,
            model=nn.Linear(2, 2),
            map_location="cpu",
        )


def test_checkpoint_from_pre_fix_framework_version_is_not_exact_resumable(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 2)
    checkpoint = manager(tmp_path, framework_version="0.1.0").save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=torch.optim.AdamW(source.parameters()),
        objective=Stateful(),
        stream=Stateful(),
    )
    with pytest.raises(CheckpointError, match="framework_version"):
        manager(tmp_path, framework_version=__version__).load_model_only(
            checkpoint,
            model=nn.Linear(2, 2),
            map_location="cpu",
        )


def test_full_resume_accepts_an_explicit_exact_legacy_run_digest(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 2)
    checkpoint = manager(tmp_path).save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=torch.optim.AdamW(source.parameters()),
        objective=Stateful({"objective": 1}),
        stream=Stateful({"cursor": 1}),
    )
    target = nn.Linear(2, 2)
    result = manager(
        tmp_path,
        run_digest="d" * 64,
        compatible_run_digests=("b" * 64,),
    ).load(
        checkpoint,
        model=target,
        optimizer=torch.optim.AdamW(target.parameters()),
        objective=Stateful(),
        stream=Stateful(),
        map_location="cpu",
    )
    assert result == (1, 0)


def _rewrite_as_distributed_runtime(
    checkpoint: Path,
    *,
    objective_states: tuple[dict, ...],
) -> None:
    runtime_path = checkpoint / "runtime.pt"
    runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
    rank_states = []
    for objective_state in objective_states:
        rank_state = {
            key: value
            for key, value in runtime.items()
            if key
            in {
                "scheduler",
                "objective",
                "stream",
                "scaler",
                "rng",
                "runtime_metadata",
            }
        }
        rank_state["objective"] = objective_state
        rank_states.append(rank_state)
    torch.save(
        {
            "identity": runtime["identity"],
            "distributed_world_size": len(rank_states),
            "rank_states": rank_states,
            "runtime_metadata": runtime["runtime_metadata"],
        },
        runtime_path,
    )
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_sha256"] = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_single_process_model_only_load_accepts_rank_invariant_objective_state(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 2)
    checkpoint = manager(tmp_path).save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=torch.optim.AdamW(source.parameters()),
        objective=Stateful({"temperature": torch.tensor(2.0)}),
        stream=Stateful(),
    )
    _rewrite_as_distributed_runtime(
        checkpoint,
        objective_states=(
            {"temperature": torch.tensor(2.0)},
            {"temperature": torch.tensor(2.0)},
        ),
    )

    objective = Stateful()
    manager(tmp_path).load_model_only(
        checkpoint,
        model=nn.Linear(2, 2),
        map_location="cpu",
        objective=objective,
    )
    assert torch.equal(objective.state["temperature"], torch.tensor(2.0))


def test_single_process_model_only_load_rejects_rank_dependent_objective_state(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 2)
    checkpoint = manager(tmp_path).save(
        global_step=1,
        micro_step=0,
        model=source,
        optimizer=torch.optim.AdamW(source.parameters()),
        objective=Stateful({"temperature": 2}),
        stream=Stateful(),
    )
    _rewrite_as_distributed_runtime(
        checkpoint,
        objective_states=({"temperature": 2}, {"temperature": 3}),
    )

    with pytest.raises(CheckpointError, match="rank-dependent"):
        manager(tmp_path).load_model_only(
            checkpoint,
            model=nn.Linear(2, 2),
            map_location="cpu",
            objective=Stateful(),
        )
