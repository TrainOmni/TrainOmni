"""Prove real BF16 varlen uses no quadratic LM mask and isolates samples."""

import json
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from trainomni.api.train import assemble
from trainomni.core.errors import SpecError
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.kernels.attention.varlen import BACKEND, VarlenLayout

from prepare import immutable_write

ROOT = Path(__file__).resolve().parent


class NoQuadratic(TorchDispatchMode):
    def __init__(self, tokens):
        self.tokens = tokens

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(result):
            if isinstance(tensor, torch.Tensor) and tensor.shape[-2:] == (self.tokens, self.tokens):
                raise AssertionError(f"dense packed mask allocated by {func}")
        return result


def main():
    task, assembly = assemble(task_path=ROOT / "task.yaml", allow_local_code=True, operation="train")
    model = assembly.model.cuda().bfloat16().eval()
    batch = DeviceContext("cuda:0", "bf16_true").move_batch(assembly.stream.next_batch(1))
    inputs = dict(batch.model_inputs)
    layout = VarlenLayout.from_packed(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        position_ids=inputs["position_ids"], segment_ids=inputs["packed_segment_ids"],
        cu_seqlens=inputs["packed_cu_seqlens"],
    )
    assert len(layout.lengths) >= 2 and inputs["attention_mask"].all()
    assert "packed_attention_mask" not in inputs
    with torch.inference_mode(), NoQuadratic(layout.total_tokens):
        original = model(**inputs).logits
        assert torch.isfinite(original).all()
        altered = {**inputs, "input_ids": inputs["input_ids"].clone(),
                   "vision": {**inputs["vision"], "hidden_states": inputs["vision"]["hidden_states"].clone()}}
        altered["input_ids"][0, layout.lengths[0]-2] = 10
        patches = int(inputs["vision"]["grid_thw"][0].prod())
        altered["vision"]["hidden_states"][:patches].zero_()
        changed = model(**altered).logits
        assert torch.equal(original[:, layout.lengths[0]:], changed[:, layout.lengths[0]:])
        assert not torch.equal(original[:, :layout.lengths[0]], changed[:, :layout.lengths[0]])
    forwards = []
    hook = model.vision.register_forward_pre_hook(lambda *args: forwards.append(1))
    corrupted = {**inputs, "packed_cu_seqlens": inputs["packed_cu_seqlens"].clone()}
    corrupted["packed_cu_seqlens"][0, -1] -= 1
    try:
        model(**corrupted)
    except SpecError:
        assert not forwards
    else:
        raise AssertionError("invalid layout accepted")
    finally:
        hook.remove()
    assert model.language.model.config._attn_implementation == BACKEND
    receipt = {"task_digest": task.digest, "module_lock": dict(assembly.module_lock),
               "backend": BACKEND, "lengths": list(layout.lengths), "lm_tokens": layout.total_tokens,
               "lm_padding_tokens": 0, "dense_language_mask": False,
               "cross_sample_isolation_max_abs": 0.0, "perturbed_sample_changed": True,
               "invalid_layout_vision_forwards": len(forwards)}
    immutable_write(ROOT / "evidence" / "runtime" / "varlen.json",
                    (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in receipt.items() if key != "module_lock"}))


if __name__ == "__main__":
    main()
