"""Strictly reload a real LoRA adapter and compare its logits to its checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from trainomni.api._checkpoint import load_model_checkpoint
from trainomni.api.train import assemble, load_resolved_run
from trainomni.modules.export.lora_adapter.module import load_lora_adapter
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.random import seed_everything


def forward(model, stream, device: DeviceContext) -> torch.Tensor:
    batch = device.move_batch(stream.next_batch(1))
    model.eval()
    with torch.inference_mode(), device.autocast():
        return model(**batch.model_inputs).logits.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    run = load_resolved_run(args.run)
    seed_everything(run.seed, deterministic=run.deterministic)
    task, assembly = assemble(task_path=args.task, allow_local_code=True)
    checkpoint_model, _, device, _, _ = load_model_checkpoint(
        task=task,
        assembly=assembly,
        run=run,
        checkpoint=args.checkpoint,
        restore_objective=False,
    )
    expected = forward(checkpoint_model, assembly.evaluation_stream, device)
    del checkpoint_model, assembly
    gc.collect()
    torch.cuda.empty_cache()

    seed_everything(run.seed, deterministic=run.deterministic)
    _, reloaded = assemble(task_path=args.task, allow_local_code=True)
    reload_device = DeviceContext(run.device, run.precision)
    reload_device.prepare_model(reloaded.model)
    load_lora_adapter(reloaded.model, args.artifact)
    actual = forward(reloaded.model, reloaded.evaluation_stream, reload_device)
    manifest = json.loads(
        (args.artifact / "manifest.json").read_text(encoding="utf-8")
    )
    result = {
        "equal": bool(torch.equal(expected, actual)),
        "max_abs_difference": float((expected.float() - actual.float()).abs().max()),
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual.float()).all()),
        "adapter_sha256": manifest["sha256"],
        "adapter_modules": len(manifest["modules"]),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["equal"] or not result["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
