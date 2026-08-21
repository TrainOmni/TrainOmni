"""Minimal native LoRA injection for torch.nn.Linear modules."""

from __future__ import annotations

import math
import re

import torch
from torch import nn
from torch.nn import functional

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from ..protocol import ParameterGroup, ParameterSelection
from .config import LoRAParameterConfig


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        base_output = self.base(inputs)
        adapted = functional.linear(functional.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base_output + adapted * self.scaling


def _parent_and_child(model: nn.Module, qualified_name: str):
    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


class LoRAParameterPolicy:
    def __init__(self, config: LoRAParameterConfig) -> None:
        self.config = config
        self.patterns = tuple(re.compile(pattern) for pattern in config.target_patterns)

    def apply(self, model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        candidates = tuple(
            (name, module)
            for name, module in model.named_modules()
            if name and isinstance(module, nn.Linear)
        )
        matched = [
            (name, module)
            for name, module in candidates
            if any(pattern.fullmatch(name) for pattern in self.patterns)
        ]
        if not matched:
            available = ", ".join(name for name, _ in candidates) or "<none>"
            raise SpecError(
                "LoRA target patterns matched no Linear modules; available: " + available
            )
        for name, linear in matched:
            parent, child_name = _parent_and_child(model, name)
            setattr(
                parent,
                child_name,
                LoRALinear(
                    linear,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                ),
            )
            if self.config.train_bias and linear.bias is not None:
                linear.bias.requires_grad_(True)
        trainable = []
        trainable_names = []
        frozen_names = []
        for name, parameter in model.named_parameters():
            if ".lora_a" in name or ".lora_b" in name or (
                self.config.train_bias
                and name.endswith(".base.bias")
                and parameter.requires_grad
            ):
                parameter.requires_grad_(True)
                trainable.append(parameter)
                trainable_names.append(name)
            else:
                parameter.requires_grad_(False)
                frozen_names.append(name)
        return ParameterSelection(
            groups=(
                ParameterGroup(
                    name=self.config.group_name,
                    parameters=tuple(trainable),
                    options={},
                ),
            ),
            trainable_names=tuple(trainable_names),
            frozen_names=tuple(frozen_names),
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("parameter_policy:trainomni/lora_linear@1"),
        config_type=LoRAParameterConfig,
        factory=lambda config, context: LoRAParameterPolicy(config),
        provides=CapabilitySet.of({"parameters.lora.linear"}),
        requires=CapabilitySet.of({"model.parameters"}),
    )
