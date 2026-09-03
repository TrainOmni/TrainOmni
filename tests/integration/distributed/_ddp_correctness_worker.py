from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import ForwardPlan, ForwardRequest, OutputRequirements
from trainomni.contracts.loss import LossBundle, LossTerm, ObjectiveMetric
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import CheckpointError, DataLoadingError
from trainomni.modules.objectives.protocol import ObjectiveRequirements
from trainomni.modules.parameters.full.config import FullParameterConfig
from trainomni.modules.parameters.full.module import FullParameterPolicy
from trainomni.runtime.checkpoint.manager import CheckpointManager
from trainomni.runtime.loop.engine import TrainEngine
from trainomni.specs.run import RunSpec


class ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.forward_calls = 0

    def forward(self, *, values):
        self.forward_calls += 1
        return SimpleNamespace(logits=self.weight * values)


class WeightedMeanObjective:
    def requirements(self):
        return ObjectiveRequirements(
            outputs=OutputRequirements(logits=True),
            metric_aggregations=(
                ("accuracy", "weighted_mean"),
                ("chosen_tokens", "sum"),
                ("delta", "weighted_mean"),
                ("preference_pairs", "sum"),
                ("rejected_tokens", "sum"),
                ("supervised_tokens", "sum"),
            ),
        )

    def plan(self, batch, context: ObjectiveContext):
        return ForwardPlan.single(
            ForwardRequest(
                "policy",
                batch.model_inputs,
                self.requirements().outputs,
                requires_grad=context.training,
            )
        )

    def compute(self, batch, outputs, context):
        del context
        errors = (outputs["policy"].require("logits").float() - batch.labels) ** 2
        numerator = errors.sum(dtype=torch.float32)
        denominator = torch.tensor(errors.numel(), dtype=torch.long)
        value = numerator / denominator
        return LossBundle(
            total=value,
            terms={"mse": LossTerm(value, 1.0, numerator, denominator)},
            metrics={
                "preference_pairs": ObjectiveMetric.sum(denominator.detach()),
                "supervised_tokens": ObjectiveMetric.sum(denominator.detach()),
                "chosen_tokens": ObjectiveMetric.sum(2 * denominator.detach()),
                "rejected_tokens": ObjectiveMetric.sum(3 * denominator.detach()),
                "accuracy": ObjectiveMetric.weighted_mean(
                    batch.labels.gt(0).sum(dtype=torch.float32),
                    denominator.detach(),
                ),
                "delta": ObjectiveMetric.weighted_mean(
                    batch.labels.sum(dtype=torch.float32),
                    denominator.detach(),
                ),
            },
        )

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise RuntimeError("objective state must be empty")


class RankStream:
    DATA: ClassVar = {
        0: (
            (torch.tensor([1.0]), torch.tensor([0.0])),
            (torch.tensor([2.0, 3.0]), torch.tensor([1.0, -1.0])),
        ),
        1: (
            (
                torch.tensor([4.0, 5.0, 6.0]),
                torch.tensor([2.0, 0.0, -2.0]),
            ),
            (torch.tensor([7.0]), torch.tensor([1.0])),
        ),
    }

    def __init__(self) -> None:
        self.rank = None
        self.cursor = 0

    def shard(self, *, rank, world_size, **kwargs):
        del kwargs
        if world_size != 2:
            raise RuntimeError("fixture requires exactly two ranks")
        self.rank = rank

    def next_batch(self, batch_size):
        if batch_size != 1 or self.rank is None:
            raise RuntimeError("invalid fixture stream state")
        values, labels = self.DATA[self.rank][self.cursor]
        self.cursor += 1
        return OmniBatch(
            (f"rank-{self.rank}-micro-{self.cursor}",),
            {"values": values},
            labels,
        )

    def state_dict(self):
        return {"rank": self.rank, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.rank = int(state["rank"])
        self.cursor = int(state["cursor"])

    def metrics(self):
        return {"cursor": self.cursor}


class Stateful:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise RuntimeError("state must be empty")


class RankFailingStateful(Stateful):
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def state_dict(self):
        if self.rank == 1:
            raise RuntimeError("injected rank-local objective state failure")
        return {}


def _run_spec(root: Path) -> RunSpec:
    return RunSpec.from_mapping(
        {
            "schema_version": 1,
            "name": "two-rank-global-mean",
            "seed": 4,
            "deterministic": True,
            "device": "cpu",
            "precision": "fp32",
            "max_steps": 1,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "optimizer": {
                "learning_rate": 0.05,
                "weight_decay": 0.0,
                "foreach": False,
            },
            "execution": {
                "backend": "torch_ddp",
                "process_group_backend": "gloo",
                "expected_world_size": 2,
                "timeout_seconds": 60,
            },
            "checkpoint": {
                "directory": str(root / "training"),
                "enabled": False,
            },
        }
    )


def _expected() -> tuple[float, float]:
    model = ScalarModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.05,
        weight_decay=0.0,
        foreach=False,
    )
    values = torch.cat(
        tuple(values for rows in RankStream.DATA.values() for values, _ in rows)
    )
    labels = torch.cat(
        tuple(labels for rows in RankStream.DATA.values() for _, labels in rows)
    )
    loss = ((model.weight * values - labels) ** 2).mean()
    loss.backward()
    optimizer.step()
    return float(model.weight.detach()), float(loss.detach())


def _rank_main(rank: int, world_size: int, output_text: str, port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "USE_LIBUV": "0",
        }
    )
    output = Path(output_text).resolve()
    root = output.parent
    model = ScalarModel()
    selection = FullParameterPolicy(FullParameterConfig()).apply(model)
    engine = TrainEngine(
        model=model,
        objective=WeightedMeanObjective(),
        parameter_selection=selection,
        stream=RankStream(),
        run=_run_spec(root),
        task_digest="a" * 64,
        module_lock={"fixture": "b" * 64},
    )
    process = engine.process
    rank = process.rank
    try:
        (record,) = engine.train()
        weights = [None] * process.world_size
        torch.distributed.all_gather_object(weights, float(model.weight.detach()))

        local_capture_root = root / "checkpoint-local-capture-failure"
        local_capture_manager = CheckpointManager(
            root=local_capture_root,
            task_digest="a" * 64,
            run_digest="c" * 64,
            module_lock={"fixture": "b" * 64},
            process=process,
        )
        try:
            local_capture_manager.save(
                global_step=1,
                micro_step=0,
                model=model,
                optimizer=engine.optimizer,
                objective=RankFailingStateful(rank),
                stream=Stateful(),
            )
        except CheckpointError as exc:
            local_capture_error = str(exc)
        else:
            raise AssertionError("rank-local checkpoint failure did not propagate")
        local_capture_errors = [None] * process.world_size
        torch.distributed.all_gather_object(
            local_capture_errors,
            local_capture_error,
        )
        local_capture_step_exists = local_capture_manager.step_path(1).exists()

        failing_root = root / "checkpoint-failure"
        if process.is_primary:
            (failing_root / "step-00000001").mkdir(parents=True)
        process.barrier()
        manager = CheckpointManager(
            root=failing_root,
            task_digest="a" * 64,
            run_digest="c" * 64,
            module_lock={"fixture": "b" * 64},
            process=process,
        )
        try:
            manager.save(
                global_step=1,
                micro_step=0,
                model=model,
                optimizer=engine.optimizer,
                objective=Stateful(),
                stream=Stateful(),
            )
        except CheckpointError as exc:
            checkpoint_error = str(exc)
        else:
            raise AssertionError("coordinated checkpoint failure did not propagate")
        checkpoint_errors = [None] * process.world_size
        torch.distributed.all_gather_object(checkpoint_errors, checkpoint_error)

        try:
            process.coordinate_primary(
                lambda: (_ for _ in ()).throw(OSError("identity receipt denied")),
                owner="run identity materialization",
                error_type=CheckpointError,
            )
        except CheckpointError as exc:
            materialize_error = str(exc)
        else:
            raise AssertionError("coordinated materialization failure did not propagate")
        materialize_errors = [None] * process.world_size
        torch.distributed.all_gather_object(materialize_errors, materialize_error)

        # A rank-local loader failure must be communicated before any DDP forward.
        original_stream = engine.stream
        def failing_batch(batch_size):
            if rank == 1:
                raise RuntimeError("injected rank-local loader failure")
            values, labels = RankStream.DATA[0][0]
            return OmniBatch(("healthy-rank",), {"values": values}, labels)
        engine.stream = SimpleNamespace(next_batch=failing_batch)
        before_weight = model.weight.detach().clone()
        calls_before_failure = model.forward_calls
        try:
            engine._optimizer_step()
        except DataLoadingError as exc:
            loader_error = str(exc)
        else:
            raise AssertionError("rank-local loader failure did not propagate")
        finally:
            engine.stream = original_stream
        assert engine.global_step == 1 and torch.equal(model.weight, before_weight)
        assert model.forward_calls == calls_before_failure
        loader_errors = [None] * process.world_size
        torch.distributed.all_gather_object(loader_errors, loader_error)

        if process.is_primary:
            expected_weight, expected_loss = _expected()
            output.write_text(
                json.dumps(
                    {
                        "weights": weights,
                        "expected_weight": expected_weight,
                        "loss": record.loss,
                        "expected_loss": expected_loss,
                        "objective_metrics": record.objective_metrics,
                        "local_capture_errors": local_capture_errors,
                        "local_capture_step_exists": local_capture_step_exists,
                        "checkpoint_errors": checkpoint_errors,
                        "materialize_errors": materialize_errors,
                        "loader_errors": loader_errors,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        process.barrier()
    finally:
        engine.close()


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    torch.multiprocessing.spawn(
        _rank_main,
        args=(2, str(output), port),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()
