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
from trainomni.models import (
    ModelCapabilities,
    ModelPluginManifest,
)
from trainomni.runtime import StageRunRequest, execute_stage


class DelegatedPlugin:
    manifest = ModelPluginManifest(
        plugin_id="delegated-test",
        plugin_version="1.0.0",
        capabilities=ModelCapabilities(
            modalities=frozenset({"text"}),
            content_blocks=frozenset({"text"}),
            objectives=frozenset({"prompt_only"}),
            max_media_per_sample=0,
            supports_generation=True,
            parallelism=frozenset({"single"}),
            engine_backends=frozenset({"delegated"}),
            export_formats=frozenset({"hf"}),
        ),
        component_ids=("policy",),
    )

    def build(self, config):
        raise AssertionError("delegated runtime must not build the model in core")


class DelegatedRuntimeTests(unittest.TestCase):
    def test_grpo_stage_is_securely_delegated_and_collected(self) -> None:
        helper = FRAMEWORK_ROOT / "tests" / "helpers" / "delegated_stage.py"
        fixture = (
            FRAMEWORK_ROOT / "tests" / "fixtures" / "valid" / "prompt_only.json"
        )
        stage = StageSpec(
            stage_id="grpo",
            stage_type="online_rl",
            objective="prompt_only",
            objective_impl="grpo",
            data=DataSpec(
                datasets=(
                    DatasetSpec(
                        dataset_id="prompt", uri=str(fixture), importer="canonical"
                    ),
                )
            ),
            component_policy={},
            optimization=OptimizationSpec(max_steps=1),
            engine=EngineSpec(
                backend="delegated",
                parallelism="single",
                precision="fp32",
                config={
                    "allow_external_command": True,
                    "argv": [sys.executable, str(helper)],
                    "result_json": "stage-result.json",
                    "environment": {"BACKEND_API_TOKEN": "must-not-be-persisted"},
                },
            ),
            checkpoint=CheckpointSpec(resume_level="stage_boundary"),
        )
        run = RunSpec(
            name="delegated",
            model=ModelSpec(plugin="delegated-test"),
            stage=stage,
        )
        resolved, report = resolve_run(run, DelegatedPlugin.manifest)
        self.assertTrue(report.valid)
        assert resolved is not None
        with tempfile.TemporaryDirectory() as directory:
            result = execute_stage(
                StageRunRequest(
                    resolved=resolved,
                    plugin=DelegatedPlugin(),
                    output_dir=Path(directory),
                    input_artifacts={},
                )
            )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["reward/mean"], 0.75)
            self.assertTrue((Path(directory) / "delegated-request.json").is_file())
            self.assertTrue((Path(directory) / "run-manifest.json").is_file())
            request_text = (Path(directory) / "delegated-request.json").read_text(
                encoding="utf-8"
            )
            provenance = json.loads(
                (Path(directory) / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("must-not-be-persisted", request_text)
            self.assertNotIn("must-not-be-persisted", json.dumps(provenance))
            self.assertEqual(
                provenance["config"]["run"]["stage"]["engine"]["config"][
                    "environment"
                ],
                "<redacted>",
            )


if __name__ == "__main__":
    unittest.main()
