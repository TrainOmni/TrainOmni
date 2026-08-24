"""Evaluate a real TrainOmni VLM artifact on VLMEvalKit AI2D_MINI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd
import torch
from PIL import Image

VALIDATION_ROOT = Path(__file__).resolve().parent
FRAMEWORK_VALIDATION_ROOT = VALIDATION_ROOT.parent
VLMEVALKIT_ROOT = Path(
    os.environ.get(
        "VLMEVALKIT_ROOT",
        r"D:\Codex\TrainOmniTemp\framework-upstream-references-20260821\upstreams\VLMEvalKit",
    )
).resolve()
CONFIG_PATH = VALIDATION_ROOT / "vlmevalkit-ai2d-real-config.json"
RUN_ROOT = VALIDATION_ROOT / "runs-real"
LMU_DATA_ROOT = VALIDATION_ROOT / "lmudata-real"
MODEL_NAME = "TrainOmniStage05AI2D"
DATASET_NAME = "AI2D_MINI"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_runner():
    return _load_source_module(
        "trainomni_vlmevalkit_real_run", VLMEVALKIT_ROOT / "run.py"
    )


def _find_single_new_run(before: set[Path]) -> Path:
    candidates = {
        path.parent
        for path in RUN_ROOT.glob(f"{MODEL_NAME}/*/status.json")
        if path.parent not in before
    }
    if len(candidates) != 1:
        raise RuntimeError(f"expected one new real evaluation run, found {candidates}")
    return candidates.pop()


def _metric_value(status: dict[str, Any]) -> float:
    metrics = status["datasets"][DATASET_NAME]["metrics"]
    for key, value in metrics.items():
        if key.endswith("|Overall"):
            return float(value)
    raise RuntimeError(f"AI2D overall metric is missing: {metrics}")


def _dataset_tsv() -> Path:
    candidates = list(LMU_DATA_ROOT.rglob(f"{DATASET_NAME}.tsv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one downloaded AI2D TSV, found {candidates}")
    return candidates[0]


def _register_model_adapter():
    import vlmeval.vlm
    from transformers import AutoProcessor, AutoTokenizer
    from vlmeval.vlm.base import BaseModel

    class TrainOmniAI2DModel(BaseModel):
        """Thin VLMEvalKit adapter for a hash-pinned TrainOmni artifact.

        AI2D is an A-D multiple-choice benchmark. The adapter performs a single
        real multimodal forward and chooses the highest-probability next token
        among the four tokenizer-native option labels. This makes the decoding
        rule deterministic while leaving visual and language scoring entirely to
        the trained model.
        """

        last_instance: ClassVar[TrainOmniAI2DModel | None] = None

        def __init__(
            self,
            *,
            artifact: str,
            artifact_sha256: str,
            vision_checkpoint: str,
            language_checkpoint: str,
            device: str,
            max_text_tokens: int,
            max_new_tokens: int,
            decode_strategy: str,
        ) -> None:
            super().__init__()
            if not torch.cuda.is_available():
                raise RuntimeError("real AI2D evaluation requires CUDA")
            self.device = torch.device(device)
            self.max_text_tokens = max_text_tokens
            self.max_new_tokens = max_new_tokens
            if decode_strategy not in {"greedy", "constrained_labels"}:
                raise ValueError(
                    "decode_strategy must be greedy or constrained_labels"
                )
            self.decode_strategy = decode_strategy
            self.calls = 0
            self.forward_calls = 0
            self.prediction_counts: dict[str, int] = {}

            artifact_path = Path(artifact).resolve()
            manifest_path = artifact_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            observed = manifest.get("sha256")
            if observed != artifact_sha256:
                raise RuntimeError(
                    f"artifact identity mismatch: expected {artifact_sha256}, got {observed}"
                )

            model_module = _load_source_module(
                "_validation_ai2d_model",
                FRAMEWORK_VALIDATION_ROOT
                / "modules"
                / "qwen_minicpm_model"
                / "module.py",
            )
            self.model = model_module._factory(
                model_module.Config(
                    vision_checkpoint=vision_checkpoint,
                    language_checkpoint=language_checkpoint,
                    load_dtype="bf16",
                    initial_artifact=str(artifact_path),
                    initial_artifact_sha256=artifact_sha256,
                ),
                SimpleNamespace(task_root=FRAMEWORK_VALIDATION_ROOT),
            )
            self.model.to(self.device)
            self.model.eval()

            processor = AutoProcessor.from_pretrained(
                vision_checkpoint, local_files_only=True
            )
            self.image_processor = processor.image_processor
            self.image_processor.size = {
                "shortest_edge": 65536,
                "longest_edge": 262144,
            }
            self.tokenizer = AutoTokenizer.from_pretrained(
                language_checkpoint, local_files_only=True
            )
            label_tokens = {
                label: self.tokenizer(label, add_special_tokens=False)["input_ids"]
                for label in "ABCD"
            }
            if any(len(tokens) != 1 for tokens in label_tokens.values()):
                raise RuntimeError(f"option labels must each be one token: {label_tokens}")
            self.label_token_ids = {
                label: tokens[0] for label, tokens in label_tokens.items()
            }
            self.artifact = artifact_path
            self.artifact_manifest = manifest
            self.loaded_cuda_bytes = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            type(self).last_instance = self

        def generate_inner(self, message, dataset=None):
            if dataset != DATASET_NAME:
                raise RuntimeError(f"adapter only supports {DATASET_NAME}, got {dataset}")
            images = [item["value"] for item in message if item["type"] == "image"]
            texts = [item["value"] for item in message if item["type"] == "text"]
            if len(images) != 1 or len(texts) != 1:
                raise RuntimeError(f"expected one image and one prompt, got {message!r}")
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": texts[0]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_tokens,
                return_tensors="pt",
            )
            image_path = Path(images[0])
            with Image.open(image_path) as opened:
                image = opened.convert("RGB").copy()
            visual = self.image_processor(images=[image], return_tensors="pt")
            inputs = {
                "input_ids": encoded["input_ids"].to(self.device),
                "attention_mask": encoded["attention_mask"].to(self.device),
                "pixel_values": visual["pixel_values"].to(self.device),
                "image_grid_thw": visual["image_grid_thw"].to(self.device),
                "image_counts": torch.tensor([1], dtype=torch.long, device=self.device),
            }
            generated: list[int] = []
            labels = tuple(self.label_token_ids)
            for _ in range(self.max_new_tokens):
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    next_token_logits = self.model(**inputs).logits[0, -1].float()
                self.forward_calls += 1
                if self.decode_strategy == "constrained_labels":
                    label_ids = torch.tensor(
                        [self.label_token_ids[label] for label in labels],
                        device=self.device,
                    )
                    scores = next_token_logits.index_select(0, label_ids)
                    prediction = labels[int(scores.argmax().item())]
                    break

                next_token = int(next_token_logits.argmax().item())
                generated.append(next_token)
                if next_token == self.tokenizer.eos_token_id:
                    break
                token = torch.tensor([[next_token]], device=self.device)
                inputs["input_ids"] = torch.cat((inputs["input_ids"], token), dim=1)
                inputs["attention_mask"] = torch.cat(
                    (
                        inputs["attention_mask"],
                        torch.ones_like(token, dtype=inputs["attention_mask"].dtype),
                    ),
                    dim=1,
                )
            else:
                prediction = self.tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()
            if self.decode_strategy == "greedy":
                prediction = self.tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()
            self.calls += 1
            self.prediction_counts[prediction] = (
                self.prediction_counts.get(prediction, 0) + 1
            )
            return prediction

    vlmeval.vlm.TrainOmniAI2DModel = TrainOmniAI2DModel
    return TrainOmniAI2DModel


def main() -> int:
    LMU_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["LMUData"] = str(LMU_DATA_ROOT)
    os.environ.setdefault("PRED_FORMAT", "tsv")
    adapter_class = _register_model_adapter()
    before = {
        path.parent for path in RUN_ROOT.glob(f"{MODEL_NAME}/*/status.json")
    }
    upstream_run = _load_upstream_runner()
    original_argv = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = [
            str(VLMEVALKIT_ROOT / "run.py"),
            "--config",
            str(CONFIG_PATH),
            "--work-dir",
            str(RUN_ROOT),
            "--mode",
            "all",
            "--judge",
            "exact_matching",
            "--judge-api-nproc",
            "1",
            "--debug",
        ]
        upstream_run.main()
    finally:
        sys.argv = original_argv
    elapsed = time.perf_counter() - started

    run_dir = _find_single_new_run(before)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    dataset_status = status.get("datasets", {}).get(DATASET_NAME, {})
    if dataset_status.get("status") != "done" or dataset_status.get("error_message"):
        raise RuntimeError(f"VLMEvalKit evaluation failed: {dataset_status}")
    instance = adapter_class.last_instance
    if instance is None:
        raise RuntimeError("VLMEvalKit did not construct the real model adapter")

    tsv_path = _dataset_tsv()
    dataset_rows = len(pd.read_csv(tsv_path, sep="\t"))
    if instance.calls != dataset_rows:
        raise RuntimeError(
            f"expected {dataset_rows} real model calls, observed {instance.calls}"
        )
    prediction_files = list(run_dir.glob(f"{MODEL_NAME}_{DATASET_NAME}.tsv"))
    score_files = list(run_dir.glob(f"{MODEL_NAME}_{DATASET_NAME}*_acc.csv"))
    if len(prediction_files) != 1 or not score_files:
        raise RuntimeError("VLMEvalKit prediction or score artifact is missing")

    receipt = {
        "status": "passed",
        "benchmark": {
            "name": DATASET_NAME,
            "rows": dataset_rows,
            "tsv": str(tsv_path),
            "tsv_sha256": _sha256(tsv_path),
            "judge": "exact_matching",
            "overall_accuracy": _metric_value(status),
        },
        "model": {
            "name": MODEL_NAME,
            "artifact": str(instance.artifact),
            "artifact_sha256": instance.artifact_manifest["sha256"],
            "artifact_file_sha256": _sha256(instance.artifact / "model.safetensors"),
            "vision_checkpoint": r"D:\Models\VLM\Qwen3.5-0.8B",
            "language_checkpoint": r"D:\Models\LLM\MiniCPM5-1B",
            "device": str(instance.device),
            "dtype": "bf16",
        },
        "inference": {
            "strategy": instance.decode_strategy,
            "calls": instance.calls,
            "forward_calls": instance.forward_calls,
            "max_new_tokens": instance.max_new_tokens,
            "label_token_ids": instance.label_token_ids,
            "prediction_counts": instance.prediction_counts,
            "loaded_cuda_bytes": instance.loaded_cuda_bytes,
            "peak_inference_cuda_bytes": torch.cuda.max_memory_allocated(instance.device),
            "elapsed_seconds": elapsed,
        },
        "vlmevalkit": {
            "root": str(VLMEVALKIT_ROOT),
            "commit": "e8e78f05f3080fe28154f2130321f17951c3be94",
            "run_dir": str(run_dir),
            "status": status,
            "prediction_file": str(prediction_files[0]),
            "score_files": [str(path) for path in score_files],
        },
    }
    receipt_path = run_dir / "trainomni-ai2d-real-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
