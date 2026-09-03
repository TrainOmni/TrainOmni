"""Real-model architecture, nested boundary and packed isolation oracle.

Reads this task only; writes a compact immutable receipt, no model payload.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from trainomni.api.train import assemble
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.kernels.attention import apply_attention_kernel

from prepare import immutable_write

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("bf16_true", "fp32"), default="bf16_true")
    args = parser.parse_args()
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260904)
    task, assembly = assemble(task_path=ROOT / "task.yaml", allow_local_code=True, operation="train")
    model = assembly.model.to(device="cuda:0", dtype=dtype).eval()
    apply_attention_kernel(model, "eager")
    device = DeviceContext("cuda:0", args.precision)
    for _ in range(10):
        batch = device.move_batch(assembly.stream.next_batch(2))
        counts = batch.model_inputs["vision"]["image_counts"]
        if bool((counts > 1).any()) and bool((counts == 1).any()):
            break
    else:
        raise AssertionError("fixture did not exercise mixed one/two-image samples")
    # Input pixel tensors are cast inside the raw encoder, not in the data workers.
    keys = tuple(model.state_dict())
    assert not any(key.startswith("vision.") and "merger" in key for key in keys)
    assert set(model.connector.state_dict()) == {
        "merger.norm.weight", "merger.norm.bias", "merger.linear_fc1.weight",
        "merger.linear_fc1.bias", "merger.linear_fc2.weight", "merger.linear_fc2.bias",
    }
    fc1, fc2 = model.connector.merger.linear_fc1, model.connector.merger.linear_fc2
    assert tuple(fc1.weight.shape) == (3072, 3072)
    assert tuple(fc2.weight.shape) == (1536, 3072)
    assert not torch.count_nonzero(fc1.bias) and not torch.count_nonzero(fc2.bias)
    paths = json.loads((ROOT / "paths.local.json").read_text())
    root = (ROOT / paths["vision_model"]).resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_file = root / index["weight_map"]["model.visual.merger.linear_fc1.weight"]
    with safe_open(weight_file, framework="pt") as weights:
        pretrained = weights.get_tensor("model.visual.merger.linear_fc1.weight")
        assert not torch.equal(pretrained.to(fc1.weight), fc1.weight)
        checked = 0
        for name, value in model.vision.model.state_dict().items():
            reference = weights.get_tensor("model.visual." + name).to(value)
            assert torch.equal(reference, value), name
            checked += 1
    inputs = dict(batch.model_inputs)
    assert set(inputs["vision"]) == {"hidden_states", "grid_thw", "image_counts"}
    assert not {"pixel_values", "image_grid_thw", "mm_token_type_ids"}.intersection(inputs)
    with torch.inference_mode():
        features = model.connector(model.vision(inputs["vision"]))
        assert features.embeddings.shape[:2] == inputs["modal_positions"].shape
        assert torch.equal(features.mask.sum(1), (inputs["modal_positions"] >= 0).sum(1))
        output = model(**inputs).logits
        assert torch.isfinite(output).all()
    receipt = {
        "task_digest": task.digest, "module_lock": dict(assembly.module_lock),
        "architecture": "pretrained raw Qwen ViT -> random full Qwen merger -> pretrained MiniCPM5",
        "raw_vit_tensors_bit_equal_to_checkpoint": checked,
        "pretrained_merger_excluded": True, "extra_linear_projection_absent": True,
        "merger_fc1_shape": list(fc1.weight.shape), "merger_fc2_shape": list(fc2.weight.shape),
        "fc1_differs_from_pretrained": True,
        "image_counts": inputs["vision"]["image_counts"].tolist(),
        "grid_thw": inputs["vision"]["grid_thw"].tolist(),
        "modal_tokens_per_pack": features.mask.sum(1).tolist(),
        "modal_embeddings_shape": list(features.embeddings.shape),
        "input_ids_shape": list(inputs["input_ids"].shape),
        "finite_forward": True,
        "precision": args.precision,
    }
    if "packed_segment_ids" in inputs:
        receipt.update(packing_oracle(model, inputs, output))
    immutable_write(ROOT / "evidence" / "runtime" / f"model-oracle-mixed-images-{args.precision}.json",
                    (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in receipt.items() if key != "module_lock"}))


def packing_oracle(model, inputs, packed_logits):
    segments = inputs["packed_segment_ids"]
    grids = inputs["vision"]["grid_thw"]
    counts = inputs["vision"]["image_counts"]
    patch_offsets = [0, *grids.prod(-1).cumsum(0).tolist()]
    image_cursor = 0
    policy = model.attention_policy
    errors, references = [], []
    with torch.inference_mode():
        try:
            model.attention_policy = None
            packed_features = model.connector(model.vision(inputs["vision"])).embeddings
            for row in range(len(segments)):
                for seg in segments[row].unique(sorted=True).tolist():
                    if seg < 0:
                        continue
                    positions = (segments[row] == seg).nonzero().flatten()
                    start, stop = int(positions[0]), int(positions[-1]) + 1
                    nimages = int(counts[row, seg])
                    modal = inputs["modal_positions"][row]
                    modal = modal[(modal >= start) & (modal < stop)] - start
                    part = {
                        "input_ids": inputs["input_ids"][row:row+1, start:stop],
                        "attention_mask": torch.ones(1, stop-start, dtype=torch.long, device=segments.device),
                        "modal_positions": modal[None],
                        "vision": {
                            "hidden_states": inputs["vision"]["hidden_states"][patch_offsets[image_cursor]:patch_offsets[image_cursor+nimages]],
                            "grid_thw": grids[image_cursor:image_cursor+nimages],
                            "image_counts": torch.tensor([[nimages]], device=segments.device),
                        },
                    }
                    individual = model(**part).logits.float()
                    actual = packed_logits[row:row+1, start:stop].float()
                    relative = float((actual - individual).norm() / individual.norm())
                    solo_features = model.connector(model.vision(part["vision"])).embeddings
                    modal_offset = int(((inputs["modal_positions"][row] >= 0) & (inputs["modal_positions"][row] < start)).sum())
                    packed_slice = packed_features[row:row+1, modal_offset:modal_offset + solo_features.shape[1]]
                    visual_relative = float((packed_slice.float() - solo_features.float()).norm() / solo_features.float().norm())
                    print(json.dumps({"row": row, "segment": seg, "logit_relative_error": relative, "vision_relative_error": visual_relative}))
                    errors.append(relative)
                    references.append((row, start, stop))
                    image_cursor += nimages
        finally:
            model.attention_policy = policy
        # Change the first image and one token in the first example only.
        altered = {**inputs, "input_ids": inputs["input_ids"].clone(),
                   "vision": {**inputs["vision"], "hidden_states": inputs["vision"]["hidden_states"].clone()}}
        altered["vision"]["hidden_states"][:patch_offsets[1]].zero_()
        first_positions = inputs["modal_positions"][0]
        candidates = [i for i in range(references[0][1], references[0][2]) if i not in first_positions]
        altered["input_ids"][0, candidates[-2]] = 10
        changed = model(**altered).logits
        unchanged_max = 0.0
        for row, start, stop in references[1:]:
            delta = float((changed[row, start:stop].float() - packed_logits[row, start:stop].float()).abs().max())
            unchanged_max = max(unchanged_max, delta)
        assert unchanged_max == 0, unchanged_max
        row, start, stop = references[0]
        assert not torch.equal(changed[row, start:stop], packed_logits[row, start:stop])
        # FP32 is the strict semantic oracle. BF16 different-shaped upstream
        # GEMMs are measured, not incorrectly promised to be numerically exact.
        if packed_logits.dtype == torch.float32:
            assert max(errors) < 1e-4, errors
    return {
        "packed_mask_shape": list(inputs["packed_attention_mask"].shape),
        "per_sample_oracle_relative_errors": errors,
        "cross_sample_isolation_max_abs": unchanged_max,
        "perturbed_sample_changed": True, "compared_samples": len(references),
    }


if __name__ == "__main__":
    main()
