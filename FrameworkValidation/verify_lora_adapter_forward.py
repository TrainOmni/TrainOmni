"""Strictly load an adapter and execute its task Objective without a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from trainomni.api.train import assemble, load_resolved_run
from trainomni.core.context import ObjectiveContext
from trainomni.modules.export.lora_adapter.module import load_lora_adapter
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.loop.step import execute_forward_plan
from trainomni.runtime.random import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    run = load_resolved_run(args.run)
    seed_everything(run.seed, deterministic=run.deterministic)
    _, assembly = assemble(task_path=args.task, allow_local_code=True)
    device = DeviceContext(run.device, run.precision)
    device.prepare_model(assembly.model)
    load_lora_adapter(assembly.model, args.artifact)
    batch = device.move_batch(assembly.evaluation_stream.next_batch(1))
    assembly.model.eval()
    with torch.inference_mode():
        loss = execute_forward_plan(
            model=assembly.model,
            objective=assembly.objective,
            batch=batch,
            context=ObjectiveContext(global_step=0, micro_step=0, training=False),
            device=device,
        )
    result = {
        "finite": bool(torch.isfinite(loss.total.detach().float())),
        "loss": float(loss.total.detach().float()),
        "terms": {
            name: float(term.value.detach().float())
            for name, term in loss.terms.items()
        },
    }
    print(json.dumps(result, sort_keys=True))
    if not result["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
