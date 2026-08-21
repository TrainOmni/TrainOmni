"""Atomic, strict export/load for TrainOmni native Linear-LoRA adapters."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import torch

from trainomni.artifacts.lineage import file_sha256
from trainomni.contracts.artifact import ArtifactIdentity
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.parameters.lora.module import LoRALinear

from .config import LoRAAdapterExportConfig


def _adapters(model):
    return tuple(
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, LoRALinear)
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class LoRAAdapterExporter:
    def __init__(self, config: LoRAAdapterExportConfig) -> None:
        self.config = config

    def export(self, *, model, destination: Path, identity, processor=None):
        del processor
        adapters = _adapters(model)
        if not adapters:
            raise SpecError("LoRA adapter export found no native LoRALinear modules")
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise SpecError("LoRA adapter export requires safetensors") from exc
        tensors = {}
        modules = {}
        for name, module in adapters:
            tensors[f"{name}.lora_a"] = module.lora_a.detach().cpu().contiguous().clone()
            tensors[f"{name}.lora_b"] = module.lora_b.detach().cpu().contiguous().clone()
            bias_key = None
            if (
                self.config.include_trainable_bias
                and module.base.bias is not None
                and module.base.bias.requires_grad
            ):
                bias_key = f"{name}.base.bias"
                tensors[bias_key] = module.base.bias.detach().cpu().contiguous().clone()
            modules[name] = {
                "rank": module.rank,
                "alpha": module.alpha,
                "in_features": module.base.in_features,
                "out_features": module.base.out_features,
                "dtype": str(module.lora_a.dtype),
                "bias_key": bias_key,
                "base_weight_sha256": _tensor_sha256(module.base.weight),
                "base_bias_sha256": (
                    None
                    if module.base.bias is None
                    else _tensor_sha256(module.base.bias)
                ),
            }
        destination = Path(destination)
        if destination.exists():
            raise SpecError(f"refusing to overwrite export: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        try:
            output = staging / self.config.filename
            save_file(tensors, output, metadata=dict(identity))
            digest = file_sha256(output)
            manifest = {
                "schema_version": 1,
                "kind": "trainomni_linear_lora",
                "file": self.config.filename,
                "sha256": digest,
                "identity": dict(sorted(identity.items())),
                "modules": modules,
            }
            temporary = staging / ".manifest.json.tmp"
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, staging / "manifest.json")
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ArtifactIdentity(
            kind="trainomni_linear_lora",
            uri=str(destination.resolve()),
            digest=digest,
        )


def load_lora_adapter(model, artifact: str | Path) -> None:
    artifact = Path(artifact)
    manifest_path = artifact / "manifest.json"
    if not manifest_path.is_file():
        raise SpecError(f"LoRA adapter manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read LoRA adapter manifest: {exc}") from exc
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "trainomni_linear_lora":
        raise SpecError("unsupported LoRA adapter manifest")
    filename = manifest.get("file")
    digest = manifest.get("sha256")
    modules = manifest.get("modules")
    if not isinstance(filename, str) or not isinstance(digest, str) or not isinstance(modules, dict):
        raise SpecError("LoRA adapter manifest is incomplete")
    tensor_path = artifact / filename
    if not tensor_path.is_file() or file_sha256(tensor_path) != digest:
        raise SpecError("LoRA adapter tensor digest mismatch")
    targets = dict(_adapters(model))
    if set(targets) != set(modules):
        raise SpecError(
            "LoRA adapter targets differ: "
            f"artifact={sorted(modules)}, model={sorted(targets)}"
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SpecError("LoRA adapter loading requires safetensors") from exc
    tensors = load_file(tensor_path)
    expected_keys = set()
    with torch.no_grad():
        for name, module in targets.items():
            metadata = modules[name]
            expected = {
                "rank": module.rank,
                "alpha": module.alpha,
                "in_features": module.base.in_features,
                "out_features": module.base.out_features,
                "dtype": str(module.lora_a.dtype),
                "base_weight_sha256": _tensor_sha256(module.base.weight),
                "base_bias_sha256": (
                    None
                    if module.base.bias is None
                    else _tensor_sha256(module.base.bias)
                ),
            }
            differences = {
                key: {"artifact": metadata.get(key), "model": value}
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if differences:
                raise SpecError(
                    f"LoRA adapter metadata differs for target {name!r}: "
                    f"{differences}"
                )
            for suffix, parameter in (("lora_a", module.lora_a), ("lora_b", module.lora_b)):
                key = f"{name}.{suffix}"
                expected_keys.add(key)
                value = tensors.get(key)
                if value is None or value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise SpecError(f"LoRA adapter tensor {key!r} is incompatible")
                parameter.copy_(value.to(parameter.device))
            bias_key = metadata.get("bias_key")
            if bias_key is not None:
                expected_keys.add(bias_key)
                if module.base.bias is None:
                    raise SpecError(f"LoRA adapter target {name!r} has no base bias")
                value = tensors.get(bias_key)
                if value is None or value.shape != module.base.bias.shape:
                    raise SpecError(f"LoRA adapter bias {bias_key!r} is incompatible")
                module.base.bias.copy_(value.to(module.base.bias.device))
    if set(tensors) != expected_keys:
        raise SpecError("LoRA adapter contains missing or unknown tensors")


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("exporter:trainomni/lora_adapter@1"),
        config_type=LoRAAdapterExportConfig,
        factory=lambda config, context: LoRAAdapterExporter(config),
        provides=CapabilitySet.of({"export.lora_adapter"}),
        requires=CapabilitySet.of({"model.parameters", "parameters.lora.linear"}),
    )
