from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pydantic import ValidationError

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.cli import main as cli_main
from trainomni.contracts import ArtifactManifest, ArtifactRef
from trainomni.recipes import (
    ArtifactCatalog,
    PipelineRuntimeState,
    PipelineSpec,
    evaluate_gate,
    load_pipeline_spec,
    topological_order,
)

PIPELINE = FRAMEWORK_ROOT / "configs" / "examples" / "toy_pipeline.yaml"
PLUGIN = FRAMEWORK_ROOT / "tests" / "plugins" / "toy_vlm_plugin.py"
PLUGIN_SPEC = f"{PLUGIN}:PLUGIN"


class PipelineTests(unittest.TestCase):
    def test_pipeline_topology_and_runtime_readiness(self) -> None:
        spec = load_pipeline_spec(PIPELINE)
        self.assertEqual(topological_order(spec), ("align", "sft"))
        state = PipelineRuntimeState.initial(spec)
        self.assertEqual(state.ready_stages(spec), ("align",))
        state = state.transition("align", "running").transition("align", "succeeded")
        self.assertEqual(state.ready_stages(spec), ("sft",))
        with self.assertRaisesRegex(ValueError, "invalid stage transition"):
            state.transition("align", "running")

    def test_pipeline_rejects_cycle(self) -> None:
        value = load_pipeline_spec(PIPELINE).model_dump(mode="json")
        value["edges"].append(
            {"from_stage": "sft", "to_stage": "align", "input_slot": "model"}
        )
        with self.assertRaisesRegex(ValidationError, "cycle"):
            PipelineSpec.model_validate(value)

    def test_gates_are_explicit_and_deterministic(self) -> None:
        passed = evaluate_gate(
            {"type": "metric", "metric": "eval_loss", "op": "le", "value": 1.0},
            metrics={"eval_loss": 0.5},
            artifacts=set(),
        )
        self.assertTrue(passed.passed)
        missing = evaluate_gate(
            {"type": "artifact", "artifact": "exported"},
            metrics={},
            artifacts={"checkpoint"},
        )
        self.assertFalse(missing.passed)
        manual = evaluate_gate(
            {"type": "manual", "approved": False, "reason": "review"},
            metrics={},
            artifacts=set(),
        )
        self.assertFalse(manual.passed)

    def test_artifact_catalog_requires_registered_lineage(self) -> None:
        catalog = ArtifactCatalog()
        base = ArtifactManifest(
            artifact_id="base",
            artifact_type="checkpoint",
            run_id="run",
            stage_id="bootstrap",
            fingerprint="a" * 64,
            resume_level="weights_only",
        )
        catalog.register(base)
        child = ArtifactManifest(
            artifact_id="aligned",
            artifact_type="checkpoint",
            run_id="run",
            stage_id="align",
            fingerprint="b" * 64,
            resume_level="exact",
            parents=(ArtifactRef("base"),),
        )
        catalog.register(child)
        self.assertEqual(catalog.resolve(ArtifactRef("aligned")), child)
        orphan = ArtifactManifest(
            artifact_id="orphan",
            artifact_type="checkpoint",
            run_id="run",
            stage_id="sft",
            fingerprint="c" * 64,
            resume_level="exact",
            parents=(ArtifactRef("missing"),),
        )
        with self.assertRaisesRegex(ValueError, "not registered"):
            catalog.register(orphan)

    def test_cli_plan_resolves_stage_dag(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--plugin",
                    PLUGIN_SPEC,
                    "--json",
                    "plan",
                    str(PIPELINE),
                ]
            )
        self.assertEqual(exit_code, 0, stdout.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["order"], ["align", "sft"])
        self.assertEqual(len(payload["fingerprint"]), 64)
        self.assertEqual(payload["edges"][0]["selector"], "best")


if __name__ == "__main__":
    unittest.main()
