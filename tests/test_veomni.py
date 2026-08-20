from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.config import (
    CheckpointSpec,
    DatasetSpec,
    DataSpec,
    EngineSpec,
    ModelSpec,
    OptimizationSpec,
    RunSpec,
    StageSpec,
    resolve_run,
)
from trainomni.engines import (
    VEOMNI_BRIDGE_API_VERSION,
    VeOmniCommandEngine,
)
from trainomni.models import ModelCapabilities, ModelPluginManifest
from trainomni.runtime import (
    StageExecutionError,
    StageRunRequest,
    execute_stage,
)


class VeOmniPlugin:
    manifest = ModelPluginManifest(
        plugin_id="veomni-test-vlm",
        plugin_version="1.0.0",
        capabilities=ModelCapabilities(
            modalities=frozenset({"text"}),
            content_blocks=frozenset({"text"}),
            objectives=frozenset({"sft"}),
            max_media_per_sample=0,
            supports_generation=False,
            parallelism=frozenset({"single"}),
            engine_backends=frozenset({"veomni"}),
            export_formats=frozenset({"hf"}),
        ),
        component_ids=("language_model",),
    )

    def build(self, config):
        raise AssertionError("VeOmni delegated bridge must not build the model in core")


def make_stage(*, revision: str = "v0.1.11", resume_level: str = "stage_boundary"):
    helper = FRAMEWORK_ROOT / "tests" / "helpers" / "delegated_stage.py"
    fixture = FRAMEWORK_ROOT / "tests" / "fixtures" / "datasets" / "torch_toy_sft.jsonl"
    return StageSpec(
        stage_id="veomni-sft",
        stage_type="instruction_sft",
        objective="sft",
        objective_impl="masked-causal-lm",
        data=DataSpec(
            datasets=(
                DatasetSpec(
                    dataset_id="sft",
                    uri=str(fixture),
                    importer="canonical",
                ),
            ),
            modalities=frozenset({"text"}),
            content_blocks=frozenset({"text"}),
            max_media_per_sample=0,
        ),
        optimization=OptimizationSpec(max_steps=1),
        engine=EngineSpec(
            backend="veomni",
            parallelism="single",
            precision="bf16",
            config={
                "allow_external_command": True,
                "argv": [sys.executable, str(helper)],
                "bridge_api": VEOMNI_BRIDGE_API_VERSION,
                "backend_revision": revision,
            },
        ),
        checkpoint=CheckpointSpec(resume_level=resume_level),
    )


def resolve(stage: StageSpec):
    run = RunSpec(
        name="veomni-bridge",
        model=ModelSpec(plugin=VeOmniPlugin.manifest.plugin_id),
        stage=stage,
    )
    resolved, report = resolve_run(run, VeOmniPlugin.manifest)
    assert report.valid, report.issues
    assert resolved is not None
    return resolved


class VeOmniBridgeTests(unittest.TestCase):
    def test_bridge_requires_immutable_revision(self) -> None:
        report = VeOmniCommandEngine().validate(make_stage(revision="main"), None)
        self.assertFalse(report.valid)
        self.assertIn(
            "engine.veomni.unpinned_revision",
            {issue.code for issue in report.issues},
        )

    def test_exact_resume_is_not_claimed_before_conformance(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(StageExecutionError, "resume_level"),
        ):
            execute_stage(
                StageRunRequest(
                    resolved=resolve(make_stage(resume_level="exact")),
                    plugin=VeOmniPlugin(),
                    output_dir=Path(directory),
                    input_artifacts={},
                )
            )

    def test_pinned_vlm_stage_executes_versioned_bridge_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = execute_stage(
                StageRunRequest(
                    resolved=resolve(make_stage()),
                    plugin=VeOmniPlugin(),
                    output_dir=output,
                    input_artifacts={},
                )
            )
            self.assertEqual(result.status, "succeeded")
            request = json.loads(
                (output / "delegated-request.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                request["backend_contract"],
                {
                    "schema_version": VEOMNI_BRIDGE_API_VERSION,
                    "engine": "veomni",
                    "backend_revision": "v0.1.11",
                    "result_contract": "trainomni.delegated-result.v1",
                },
            )
            self.assertEqual(request["resolved"]["run"]["stage"]["objective"], "sft")
            manifest = json.loads(
                (output / "run-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["engine"]["id"], "veomni")


if __name__ == "__main__":
    unittest.main()
