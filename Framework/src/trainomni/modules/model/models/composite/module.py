"""Composite encoder -> connector -> fusion -> language model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from trainomni.contracts.features import (
    ModalFeatureBranch,
    ModalFeatures,
    ModalFeatureSet,
)
from trainomni.core.capability import CapabilitySet
from trainomni.core.context import BuildContext
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import CompositeBranchConfig, CompositeModelConfig


class CompositeModel(nn.Module):
    def __init__(
        self,
        *,
        branches: tuple[CompositeBranchConfig, ...] | None = None,
        components: Mapping[str, nn.Module] | None = None,
        encoder: nn.Module | None = None,
        connector: nn.Module | None = None,
        fusion: nn.Module | None = None,
        language: nn.Module | None = None,
        fusion_name: str = "fusion",
        language_name: str = "language",
        attention_policy: Any | None = None,
    ):
        super().__init__()
        if components is None:
            if any(item is None for item in (encoder, connector, fusion, language)):
                raise ValueError(
                    "legacy composite construction requires encoder, connector, fusion "
                    "and language"
                )
            components = {
                "encoder": encoder,
                "connector": connector,
                "fusion": fusion,
                "language": language,
            }
        elif any(item is not None for item in (encoder, connector, fusion, language)):
            raise ValueError("components cannot be combined with legacy component arguments")
        if branches is None:
            branches = CompositeModelConfig().branches
        self.branches = tuple(branches)
        self._fusion_name = fusion_name
        self._language_name = language_name
        required_components = tuple(
            dict.fromkeys(
                (
                    *(branch.encoder for branch in self.branches),
                    *(branch.connector for branch in self.branches),
                    fusion_name,
                    language_name,
                )
            )
        )
        missing = sorted(set(required_components) - set(components))
        if missing:
            raise ValueError("composite components are missing: " + ", ".join(missing))
        for name in required_components:
            if name in {"attention_policy", "branches"} or hasattr(self, name):
                raise ValueError(f"composite component name is reserved: {name}")
            self.add_module(name, components[name])
        self.attention_policy = attention_policy

    def _component(self, name: str) -> nn.Module:
        return self._modules[name]

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        modal_inputs: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        modal_positions: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        routed_inputs = dict(kwargs)
        if pixel_values is not None:
            routed_inputs["pixel_values"] = pixel_values
        if modal_positions is not None:
            routed_inputs["modal_positions"] = modal_positions
        if modal_inputs is not None:
            if len(self.branches) != 1:
                raise SpecError("modal_inputs alias is only valid for one composite branch")
            input_key = self.branches[0].input_key
            if input_key in routed_inputs:
                raise SpecError(
                    f"both modal_inputs and explicit branch input {input_key!r} were provided"
                )
            routed_inputs[input_key] = modal_inputs

        branch_outputs = []
        for branch in self.branches:
            if branch.input_key not in routed_inputs:
                if branch.positions_key is not None and branch.positions_key in routed_inputs:
                    raise SpecError(
                        f"branch {branch.name!r} has positions but no {branch.input_key!r}"
                    )
                if branch.required:
                    raise SpecError(
                        f"required branch {branch.name!r} input is missing: "
                        f"{branch.input_key}"
                    )
                continue
            encoder_inputs = routed_inputs.pop(branch.input_key)
            explicit_positions = (
                None
                if branch.positions_key is None
                else routed_inputs.pop(branch.positions_key, None)
            )
            features = self._component(branch.encoder)(encoder_inputs)
            if not isinstance(features, ModalFeatures):
                features = ModalFeatures(embeddings=features)
            connected = self._component(branch.connector)(features)
            if not isinstance(connected, ModalFeatures):
                connected = ModalFeatures(embeddings=connected)
            if explicit_positions is not None:
                if connected.positions is not None and not torch.equal(
                    connected.positions, explicit_positions
                ):
                    raise SpecError(
                        f"branch {branch.name!r} encoder positions disagree with model input"
                    )
                connected = replace(connected, positions=explicit_positions)
            branch_outputs.append(
                ModalFeatureBranch(branch.name, branch.modality, connected)
            )

        feature_set = ModalFeatureSet(tuple(branch_outputs))
        aggregate_positions = None
        if feature_set.branches:
            try:
                aggregate_positions = feature_set.concatenate().positions
            except ValueError as exc:
                raise SpecError(f"invalid composite modal branches: {exc}") from exc
        if self.attention_policy is not None:
            attention = self.attention_policy.apply(
                input_ids=input_ids,
                attention_mask=attention_mask,
                modal_positions=aggregate_positions,
                model_inputs=MappingProxyType(routed_inputs),
            )
            attention_mask = attention.attention_mask
            for field in attention.consumed_model_inputs:
                if field not in routed_inputs:
                    raise SpecError(
                        f"attention policy consumed missing model input: {field}"
                    )
                routed_inputs.pop(field)
            overlap = sorted(set(routed_inputs) & set(attention.model_kwargs))
            if overlap:
                raise SpecError(
                    "attention policy attempted to overwrite model inputs: "
                    + ", ".join(overlap)
                )
            routed_inputs.update(attention.model_kwargs)
        return self._component(self._fusion_name)(
            language=self._component(self._language_name),
            input_ids=input_ids,
            modal_features=feature_set,
            attention_mask=attention_mask,
            modal_positions=aggregate_positions,
            **routed_inputs,
        )


def _factory(config: CompositeModelConfig, context: BuildContext) -> CompositeModel:
    names = tuple(
        dict.fromkeys(
            (
                *(branch.encoder for branch in config.branches),
                *(branch.connector for branch in config.branches),
                config.fusion,
                config.language,
            )
        )
    )
    missing = sorted(set(names) - set(context.components))
    if missing:
        raise SpecError(f"composite model is missing components: {', '.join(missing)}")
    return CompositeModel(
        branches=config.branches,
        components={name: context.components[name] for name in names},
        fusion_name=config.fusion,
        language_name=config.language,
        attention_policy=context.components.get("__attention_policy__"),
    )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("model:trainomni/composite@1"),
        config_type=CompositeModelConfig,
        factory=_factory,
        provides=CapabilitySet.of(
            {
                "model.composite",
                "model.output.logits",
                "model.parameters",
                "model.attention.semantic",
            }
        ),
        requires=CapabilitySet.of(
            {"component.encoder", "component.connector", "component.fusion", "component.language"}
        ),
    )
