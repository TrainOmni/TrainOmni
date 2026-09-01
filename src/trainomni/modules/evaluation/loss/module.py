"""Denominator-correct aggregation of one named objective loss term."""

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import ObjectiveError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import LossEvaluatorConfig


class LossEvaluator:
    def __init__(self, config: LossEvaluatorConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.numerator = 0.0
        self.denominator = 0.0

    def update(self, batch, loss) -> None:
        del batch
        term = loss.terms.get(self.config.term)
        if term is None:
            raise ObjectiveError(
                f"loss evaluator cannot find term {self.config.term!r}"
            )
        self.numerator += float(term.numerator.detach().float().item())
        self.denominator += float(term.denominator.detach().float().item())

    def compute(self):
        if self.denominator <= 0:
            raise ObjectiveError("loss evaluator has no accumulated denominator")
        return {self.config.metric_name: self.numerator / self.denominator}


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("evaluator:trainomni/loss@1"),
        config_type=LossEvaluatorConfig,
        factory=lambda config, context: LossEvaluator(config),
        provides=CapabilitySet.of({"evaluation.loss"}),
        requires=CapabilitySet.of({"objective.loss_bundle"}),
    )
