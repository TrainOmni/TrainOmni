from dataclasses import dataclass

import pytest

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import CapabilityError, RegistryError, SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId, ModuleKind, ModuleRef
from trainomni.core.registry import ModuleRegistry
from trainomni.core.resolver import ModuleResolver


@dataclass(frozen=True)
class DummyConfig:
    value: int = 1


def make_descriptor(
    module_id: str,
    *,
    provides: set[str] | None = None,
    requires: set[str] | None = None,
) -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse(module_id),
        config_type=DummyConfig,
        factory=lambda config, context: (config, context),
        provides=CapabilitySet.of(provides or set()),
        requires=CapabilitySet.of(requires or set()),
    )


def test_registry_is_explicit_strict_and_typed() -> None:
    descriptor = make_descriptor("objective:example/custom@1")
    registry = ModuleRegistry([descriptor])
    reference = ModuleRef.from_mapping(
        {"module": "objective:example/custom@1", "config": {"value": 4}},
        field_name="objective",
    )
    resolved = ModuleResolver(registry).resolve(reference, kind=ModuleKind.OBJECTIVE)
    assert resolved.config == DummyConfig(value=4)

    with pytest.raises(RegistryError, match="duplicate module"):
        registry.register(descriptor)
    with pytest.raises(RegistryError, match="expected model"):
        registry.descriptor(reference, expected_kind=ModuleKind.MODEL)
    with pytest.raises(SpecError, match="unknown keys"):
        ModuleResolver(registry).resolve(
            ModuleRef.from_mapping(
                {"module": "objective:example/custom@1", "config": {"vale": 4}},
                field_name="objective",
            ),
            kind=ModuleKind.OBJECTIVE,
        )


def test_capability_preflight_happens_without_construction() -> None:
    builds = []

    @dataclass(frozen=True)
    class Config:
        pass

    descriptor = ModuleDescriptor(
        module_id=ModuleId.parse("objective:example/needs_logits@1"),
        config_type=Config,
        factory=lambda config, context: builds.append((config, context)),
        requires=CapabilitySet.of({"model.output.logits"}),
    )
    reference = ModuleRef.from_mapping(
        {"module": str(descriptor.module_id)}, field_name="objective"
    )
    resolved = ModuleResolver(ModuleRegistry([descriptor])).resolve(
        reference, kind=ModuleKind.OBJECTIVE
    )
    with pytest.raises(CapabilityError, match="model.output.logits"):
        ModuleResolver.preflight((resolved,))
    assert builds == []


def test_module_cannot_satisfy_its_own_requirement() -> None:
    descriptor = make_descriptor(
        "objective:example/self_cycle@1",
        provides={"cycle"},
        requires={"cycle"},
    )
    reference = ModuleRef.from_mapping(
        {"module": str(descriptor.module_id)}, field_name="objective"
    )
    resolver = ModuleResolver(ModuleRegistry([descriptor]))
    resolved = resolver.resolve(reference, kind=ModuleKind.OBJECTIVE)
    with pytest.raises(CapabilityError, match="cycle"):
        resolver.preflight((resolved,))
