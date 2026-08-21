"""Model-only checkpoint loading shared by evaluation and export."""

from __future__ import annotations

from pathlib import Path

from trainomni import __version__
from trainomni.runtime.checkpoint.manager import CheckpointManager
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.kernels.attention import apply_attention_kernel
from trainomni.runtime.kernels.compilation import compile_forward


def load_model_checkpoint(
    *,
    task,
    assembly,
    run,
    checkpoint: str | Path,
    restore_objective: bool,
):
    device = DeviceContext(run.device, run.precision)
    apply_attention_kernel(assembly.model, run.attention_kernel)
    device.prepare_model(assembly.model)
    manager = CheckpointManager(
        root=run.checkpoint.directory,
        task_digest=task.digest,
        run_digest=run.digest,
        module_lock=assembly.module_lock,
        framework_version=__version__,
    )
    checkpoint_path = Path(checkpoint).resolve()
    manifest = manager.load_model_only(
        checkpoint_path,
        model=assembly.model,
        map_location=device.device,
        objective=assembly.objective if restore_objective else None,
    )
    execution_model = compile_forward(assembly.model, run.compile)
    return assembly.model, execution_model, device, checkpoint_path, manifest
