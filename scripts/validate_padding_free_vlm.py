"""Read local pinned VLM fixtures, write only a compact Framework receipt.

This probe is an explicit model-specific fusion adapter, not a claim that every
VLM automatically supports varlen. No checkpoint/model/data payload is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from trainomni.contracts.forward import ForwardResult
from trainomni.core.context import BuildContext, ObjectiveContext
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleKind
from trainomni.modules.data.collation.padding_free.module import PaddingFreeCollator
from trainomni.modules.data.packing.padding_free.module import PaddingFreePacker
from trainomni.modules.objectives.causal_lm.config import CausalLMConfig
from trainomni.modules.objectives.causal_lm.module import CausalLMObjective
from trainomni.runtime.kernels.attention.varlen import VarlenLayout, padding_free_forward

ROOT = Path(__file__).resolve().parents[1]


class NoQuadraticLMAllocation(TorchDispatchMode):
    def __init__(self, tokens):
        self.tokens = tokens

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(result):
            if isinstance(tensor, torch.Tensor) and tensor.shape[-2:] == (self.tokens, self.tokens):
                raise AssertionError(f"quadratic packed-LM allocation in {func}: {tensor.shape}")
        return result


class VisualPrefixProbe(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.last_layout = None
        self.language_forwards = 0

    def forward(self, *, separate=False, **inputs):
        # Validate text metadata before any vision or language model execution.
        layout = VarlenLayout.from_packed(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            segment_ids=inputs["packed_segment_ids"],
            cu_seqlens=inputs["packed_cu_seqlens"],
        )
        counts = inputs["image_counts"].flatten().tolist()
        if len(counts) != len(layout.lengths) or any(n <= 0 for n in counts):
            raise SpecError("image counts must align with packed samples")
        if sum(counts) != inputs["image_grid_thw"].shape[0]:
            raise SpecError("image counts disagree with visual grids")
        features = self.base._vision_features(
            pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]
        )
        text = self.base.language_model.get_input_embeddings()(inputs["input_ids"])
        sequences, prefixes = [], []
        image_cursor = 0
        for part, count in zip(text.split(layout.lengths, dim=1), counts, strict=True):
            projected, image_cursor = self.base._project_images(
                features, cursor=image_cursor, count=count, dtype=text.dtype
            )
            sequences.append(torch.cat((projected[None], part), dim=1))
            prefixes.append(projected.shape[0])
        expanded = VarlenLayout(tuple(part.shape[1] for part in sequences))
        self.last_layout = {
            "text_lengths": list(layout.lengths),
            "visual_prefix_lengths": prefixes,
            "expanded_lengths": list(expanded.lengths),
            "lm_tokens": expanded.total_tokens,
            "lm_padding_tokens": 0,
        }
        if separate:
            self.base.language_model.set_attn_implementation("sdpa")
            self.language_forwards += len(sequences)
            outputs = [
                self.base.language_model(
                    inputs_embeds=part, use_cache=False, return_dict=True
                ).logits
                for part in sequences
            ]
        else:
            self.language_forwards += 1
            with NoQuadraticLMAllocation(expanded.total_tokens):
                output = padding_free_forward(
                    self.base.language_model,
                    inputs_embeds=torch.cat(sequences, dim=1),
                    layout=expanded,
                ).logits
            outputs = output.split(expanded.lengths, dim=1)
        logits = torch.cat(
            [values[:, prefix:] for values, prefix in zip(outputs, prefixes, strict=True)], dim=1
        )
        return SimpleNamespace(logits=logits)


def relative_error(actual, reference):
    return float((actual.float() - reference.float()).norm() / reference.float().norm())


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=ROOT.parent / "FrameworkValidation")
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/varlen-validation-20260903/real-vlm.json"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("verification receipts must stay within Framework")
    fixtures = args.fixtures.resolve()
    sys.dont_write_bytecode = True  # External fixture modules are read-only.
    torch.manual_seed(26090319)
    torch.backends.cuda.matmul.allow_tf32 = False
    helpers = runpy.run_path(str(fixtures / "verify_extension_routes.py"))
    task, _, stream, resolver = helpers["_data_only_stream"](
        fixtures / "extension-packing.task.json"
    )
    # Same pinned source/IO/objective/checkpoint. Only swap the data representation.
    stream.packer = PaddingFreePacker(stream.packer.config)
    stream.collator = PaddingFreeCollator(stream.collator.config)
    cpu_batch = stream.next_batch(1)
    assert cpu_batch.model_inputs["input_ids"].shape == (1, 39)
    assert "packed_attention_mask" not in cpu_batch.model_inputs
    batch = replace(
        cpu_batch,
        labels=cpu_batch.labels.cuda(),
        model_inputs={k: v.cuda() for k, v in cpu_batch.model_inputs.items()},
        supervision={k: v.cuda() for k, v in cpu_batch.supervision.items()},
    )
    base = (
        resolver.resolve(task.model.implementation, kind=ModuleKind.MODEL)
        .build(BuildContext(task_digest=task.digest, task_root=fixtures))
        .cuda()
        .bfloat16()
    )
    base.requires_grad_(False)
    base.connector.requires_grad_(True)
    model = VisualPrefixProbe(base).eval()
    objective = CausalLMObjective(CausalLMConfig())
    context = ObjectiveContext(global_step=0, micro_step=0)

    def loss(result):
        return objective.compute(batch, {"policy": ForwardResult("policy", result)}, context)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    params = tuple(base.connector.parameters())
    # Independent-sequence oracle keeps the original CE weighting/causal labels.
    expected = model(**batch.model_inputs, separate=True)
    reference_loss = loss(expected)
    reference_grads = torch.autograd.grad(reference_loss.total, params)
    reference_logits = expected.logits.detach().clone()
    reference_ce = float(reference_loss.total.detach())
    del expected, reference_loss
    actual = model(**batch.model_inputs)
    actual_loss = loss(actual)
    grads = torch.autograd.grad(actual_loss.total, params)
    logit_error = relative_error(actual.logits.detach(), reference_logits)
    grad_errors = [relative_error(g, r) for g, r in zip(grads, reference_grads, strict=True)]
    ce = float(actual_loss.total.detach())
    assert logit_error < 0.015, logit_error
    assert abs(ce - reference_ce) < 0.02, (ce, reference_ce)
    assert max(grad_errors) < 0.06, grad_errors
    max_logit_error = float((actual.logits.detach().float() - reference_logits.float()).abs().max())
    first_length = int(batch.model_inputs["packed_cu_seqlens"][0, 1])
    baseline_second = actual.logits[:, first_length:].detach().clone()
    del actual, actual_loss, reference_logits, grads, reference_grads

    # Change sample A's input; sample B must remain bit-identical at fixed shape.
    changed_inputs = dict(batch.model_inputs)
    changed_inputs["input_ids"] = changed_inputs["input_ids"].clone()
    changed_inputs["input_ids"][0, 1] = 17
    with torch.no_grad():
        isolated = model(**changed_inputs).logits[:, first_length:]
        isolation_error = float((isolated - baseline_second).abs().max())
    assert isolation_error == 0, isolation_error
    del isolated, baseline_second

    # Mutate valid-looking positions; no model forward may happen.
    corrupt = dict(batch.model_inputs)
    corrupt["packed_cu_seqlens"] = corrupt["packed_cu_seqlens"].clone()
    corrupt["packed_cu_seqlens"][0, 1] -= 1
    forwards = model.language_forwards
    try:
        model(**corrupt)
    except SpecError as exc:
        negative = str(exc)
    else:
        raise AssertionError("misaligned boundaries were accepted")
    assert model.language_forwards == forwards

    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=0, foreach=False)
    steps = []
    model.train()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in params]
            result = model(**batch.model_inputs)
            bundle = loss(result)
            bundle.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            assert torch.isfinite(norm) and norm > 0
            optimizer.step()
            changed = sum(int((p != old).sum()) for p, old in zip(params, before, strict=True))
            assert changed > 0
            steps.append(
                {
                    "step": step + 1,
                    "loss": float(bundle.total.detach()),
                    "grad_norm": float(norm),
                    "changed_connector_elements": changed,
                }
            )
            del before, result, bundle
    operators = {
        event.key: event.count
        for event in prof.key_averages()
        if "efficient_attention" in event.key
    }
    assert operators.get("aten::_efficient_attention_forward", 0) >= 48, operators
    assert operators.get("aten::_efficient_attention_backward", 0) >= 48, operators
    assert all(p.grad is None for p in base.language_model.parameters())
    assert all(p.grad is None for p in base.vision_encoder.parameters())
    model.eval()
    with torch.inference_mode():
        evaluation = float(loss(model(**batch.model_inputs)).total)
    torch.cuda.synchronize()
    import transformers
    import xformers

    receipt = {
        "status": "passed",
        "source_sha256": source_digest(),
        "task_sha256": task.digest,
        "initial_artifact": dict(task.model.implementation.config),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "xformers": xformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "backend": "xformers CUTLASS / aten efficient_attention (not FlashAttention)",
        "flash_attention_built": torch.backends.cuda.is_flash_attention_available(),
        "precision": "bf16_true",
        "layout": model.last_layout,
        "quadratic_lm_allocation_guard": "passed",
        "upstream_operators": operators,
        "oracle": {
            "reference": "separate SDPA forwards, same objective and token weighting",
            "logit_relative_l2": logit_error,
            "logit_max_abs": max_logit_error,
            "connector_grad_relative_l2": grad_errors,
            "reference_ce": reference_ce,
            "padding_free_ce": ce,
        },
        "alignment_negative": {"error": negative, "additional_language_forwards": 0},
        "cross_sample_isolation_max_abs": isolation_error,
        "steps": steps,
        "same_fixture_eval_ce": evaluation,
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "seconds_after_load": time.perf_counter() - started,
        "boundary": "Explicit local visual-prefix adapter; connector-only engineering verification. "
        "No model-quality, generic-VLM, flash-attn, distributed, or speedup claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, allow_nan=False))


if __name__ == "__main__":
    main()
