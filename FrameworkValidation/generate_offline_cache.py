"""Generate hash-pinned KD/DPO caches from one exported real VLM artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from trainomni.modules.data.sources.jsonl.config import JsonlSourceConfig
from trainomni.modules.data.sources.jsonl.module import JsonlSource
from trainomni.modules.export.safetensors.module import load_safetensors_artifact
from trainomni.runtime.device.context import DeviceContext

ROOT = Path(__file__).resolve().parent
VISION = Path(r"D:\Models\VLM\Qwen3.5-0.8B")
LANGUAGE = Path(r"D:\Models\LLM\MiniCPM5-1B")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validation module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(*, artifact: Path, mode: str, max_text_tokens: int):
    model_module = load_source_module(
        "_validation_cache_model",
        ROOT / "modules" / "qwen_minicpm_model" / "module.py",
    )
    io_module = load_source_module(
        "_validation_cache_io",
        ROOT / "modules" / "qwen_minicpm_io" / "module.py",
    )
    model = model_module._factory(
        model_module.Config(
            vision_checkpoint=str(VISION),
            language_checkpoint=str(LANGUAGE),
            load_dtype="bf16",
        ),
        SimpleNamespace(task_root=ROOT),
    )
    load_safetensors_artifact(model, artifact)
    model_io = io_module.ModelIO(
        io_module.Config(
            vision_checkpoint=str(VISION),
            language_checkpoint=str(LANGUAGE),
            mode=mode,
            max_text_tokens=max_text_tokens,
        ),
        task_root=ROOT,
    )
    device = DeviceContext("cuda:0", "bf16_true")
    device.prepare_model(model)
    model.eval()
    return model, model_io, device


def batched(inputs: dict[str, torch.Tensor], device: DeviceContext):
    values = {
        "input_ids": inputs["input_ids"].unsqueeze(0),
        "attention_mask": inputs["attention_mask"].unsqueeze(0),
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "image_counts": inputs["image_counts"],
    }
    return {name: value.to(device.device) for name, value in values.items()}


def source(path: Path, digest: str):
    return JsonlSource(path, JsonlSourceConfig(path=str(path), sha256=digest, repeat=False))


def write_sample(
    output: Path,
    index: dict,
    sample_id: str,
    tensors: dict[str, torch.Tensor],
) -> None:
    tensor_file = output / f"{sample_id}.safetensors"
    save_file(tensors, tensor_file)
    index["samples"][sample_id] = {
        "file": tensor_file.name,
        "sha256": sha256(tensor_file),
        "tensors": {name: name for name in sorted(tensors)},
    }


def finalize_index(output: Path, index: dict) -> None:
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate_kd(args) -> None:
    model, model_io, device = load_runtime(
        artifact=args.artifact,
        mode="dense_kd",
        max_text_tokens=args.max_text_tokens,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "samples": {}}
    for data_path, digest in zip(args.data, args.data_sha256, strict=True):
        stream = source(data_path, digest)
        while True:
            try:
                sample = stream.next_sample()
            except StopIteration:
                break
            prompt, answer = model_io._caption_values(sample)
            input_ids, attention_mask, _ = model_io._encode_prompt_answer(prompt, answer)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **model_io._vision_inputs(sample),
            }
            with torch.inference_mode(), device.autocast():
                logits = model(**batched(inputs, device)).logits[0]
            write_sample(
                args.output,
                index,
                sample.sample_id,
                {"teacher_logits": logits.cpu().bfloat16()},
            )
    finalize_index(args.output, index)


def branch_logps(model, model_io, device, sample, answer: str) -> torch.Tensor:
    inputs, labels = model_io._branch(sample, answer)
    with torch.inference_mode(), device.autocast():
        logits = model(**batched(inputs, device)).logits[0, :-1].float()
    targets = labels[1:].to(device.device)
    safe_targets = targets.masked_fill(targets.eq(-100), 0)
    values = torch.log_softmax(logits, dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    return values.masked_fill(targets.eq(-100), 0.0).cpu().float()


def generate_dpo(args) -> None:
    model, model_io, device = load_runtime(
        artifact=args.artifact,
        mode="preference",
        max_text_tokens=args.max_text_tokens,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "samples": {}}
    for data_path, digest in zip(args.data, args.data_sha256, strict=True):
        stream = source(data_path, digest)
        while True:
            try:
                sample = stream.next_sample()
            except StopIteration:
                break
            chosen = sample.metadata.get("chosen")
            rejected = sample.metadata.get("rejected")
            if not isinstance(chosen, str) or not isinstance(rejected, str):
                raise TypeError("DPO source requires chosen/rejected strings")
            tensors = {
                "chosen_reference_logps": branch_logps(
                    model, model_io, device, sample, chosen
                ),
                "rejected_reference_logps": branch_logps(
                    model, model_io, device, sample, rejected
                ),
            }
            write_sample(args.output, index, sample.sample_id, tensors)
    finalize_index(args.output, index)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("kd", "dpo"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--data-sha256", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-text-tokens", type=int, default=96)
    parsed = parser.parse_args()
    if len(parsed.data) != len(parsed.data_sha256):
        parser.error("--data and --data-sha256 counts must match")
    return parsed


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.kind == "kd":
        generate_kd(arguments)
    else:
        generate_dpo(arguments)
