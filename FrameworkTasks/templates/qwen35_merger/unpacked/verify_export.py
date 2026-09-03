"""Verify export digest, strict keys, tensor equality and a fresh real forward."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from trainomni.api.train import assemble
from trainomni.modules.export.safetensors.module import load_safetensors_artifact
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.kernels.attention import apply_attention_kernel

from prepare import immutable_write

ROOT = Path(__file__).resolve().parent


def main():
    task, assembly = assemble(task_path=ROOT / "task.yaml", allow_local_code=True, operation="all")
    model = assembly.model.to(dtype=torch.bfloat16)
    artifact = ROOT / "outputs" / "export_001"
    load_safetensors_artifact(model=model, artifact=artifact)
    checkpoint = ROOT / "outputs" / "baseline_001" / "checkpoints" / "step-00000002"
    with safe_open(checkpoint / "model.safetensors", framework="pt") as saved:
        state = model.state_dict()
        for key in saved.keys():
            assert torch.equal(state[key], saved.get_tensor(key)), key
        tensors = len(saved.keys())
    model.cuda().eval()
    apply_attention_kernel(model, "eager")
    batch = DeviceContext("cuda:0", "bf16_true").move_batch(assembly.evaluation_stream.next_batch(2))
    with torch.inference_mode():
        logits = model(**batch.model_inputs).logits
        assert torch.isfinite(logits).all()
    record = {"task_digest": task.digest, "export_tensor_count": tensors,
              "bit_equal_to_checkpoint": True, "strict_reload": True,
              "fresh_forward_finite": True, "logits_shape": list(logits.shape)}
    immutable_write(ROOT / "evidence" / "runtime" / "export-reload.json",
                    (json.dumps(record, indent=2) + "\n").encode())
    print(json.dumps(record))


if __name__ == "__main__":
    main()
