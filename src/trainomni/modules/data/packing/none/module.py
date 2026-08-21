"""Emit every supervised example without sequence packing."""

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import NoPackingConfig


class NoPacker:
    def add(self, sample):
        return (sample,)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise SpecError("no-packing policy has no mutable state")


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("packer:trainomni/none@1"),
        config_type=NoPackingConfig,
        factory=lambda config, context: NoPacker(),
        provides=CapabilitySet.of({"data.packed"}),
        requires=CapabilitySet.of({"data.supervised"}),
    )
