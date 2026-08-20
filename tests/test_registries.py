from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.contracts import (
    BatchBudget,
    BatchItem,
    BatchPlan,
    CostVector,
)
from trainomni.engines import (
    EngineCapabilities,
    EngineKind,
    EngineManifest,
    EngineRegistry,
    EngineRegistryError,
)
from trainomni.models import ModelBatch
from trainomni.objectives import (
    MaskedCausalLMObjective,
    ObjectiveRegistry,
    ObjectiveRegistryError,
)


@dataclass
class Output:
    loss: float


class FakeModel:
    def __call__(self, **kwargs):
        return Output(loss=2.5)


class FakeEngine:
    manifest = EngineManifest(
        engine_id="fake",
        engine_version="1.0.0",
        kind=EngineKind.LOOP,
        capabilities=EngineCapabilities(
            stage_types=frozenset({"instruction_sft"}),
            objectives=frozenset({"masked-causal-lm"}),
            parallelism=frozenset({"single"}),
            precisions=frozenset({"fp32"}),
            resume_levels=frozenset({"exact"}),
        ),
    )

    def validate(self, stage, model):
        return None

    def prepare(self, context):
        return context

    def run(self, prepared):
        return prepared

    def checkpoint(self, prepared, reason):
        return reason

    def collect(self, result):
        return result


class RegistryTests(unittest.TestCase):
    def test_masked_causal_lm_uses_named_loss_and_token_denominator(self) -> None:
        plan = BatchPlan(
            items=(BatchItem("sample", 0, CostVector(text_tokens=4)),),
            total_cost=CostVector(text_tokens=4),
            budget=BatchBudget(max_text_tokens=8),
        )
        batch = ModelBatch(
            sample_ids=("sample",),
            model_inputs={
                "input_ids": [[1, 2, 3, 4]],
                "labels": [[-100, 2, 3, 4]],
            },
            plan=plan,
        )
        objective = MaskedCausalLMObjective()
        prepared = objective.prepare(batch, None)
        output = objective.compute(FakeModel(), prepared)
        self.assertEqual(output.total, 2.5)
        self.assertEqual(output.terms["token_ce"].denominator, 3)
        self.assertEqual(output.counts["loss_tokens"], 3)

    def test_objective_registry_has_builtin_and_rejects_duplicate(self) -> None:
        registry = ObjectiveRegistry()
        self.assertEqual(registry.get("masked-causal-lm").manifest.objective_version, "1.0.0")
        with self.assertRaisesRegex(ObjectiveRegistryError, "already registered"):
            registry.register(MaskedCausalLMObjective())

    def test_engine_registry_checks_contract_and_duplicate(self) -> None:
        registry = EngineRegistry()
        self.assertEqual(
            {item.engine_id for item in registry.manifests()},
            {"torch", "delegated", "trl", "verl", "veomni", "nemo"},
        )
        engine = FakeEngine()
        registry.register(engine)
        self.assertIs(registry.get("fake"), engine)
        with self.assertRaisesRegex(EngineRegistryError, "already registered"):
            registry.register(FakeEngine())


if __name__ == "__main__":
    unittest.main()
