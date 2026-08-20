from __future__ import annotations

import importlib.util
import io
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.cli import main as cli_main
from trainomni.config import load_run_spec
from trainomni.recipes import PipelineSpec, StageEdge, resolve_pipeline
from trainomni.registry import ModelPluginRegistry
from trainomni.runtime import PipelineExecutor

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PLUGIN = FRAMEWORK_ROOT / "tests" / "plugins" / "torch_toy_vlm_plugin.py"
CONFIG = FRAMEWORK_ROOT / "configs" / "examples" / "torch_toy_smoke.yaml"


@unittest.skipUnless(TORCH_AVAILABLE, "requires optional torch runtime")
class TorchEndToEndTests(unittest.TestCase):
    def _cli(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main(["--plugin", f"{PLUGIN}:PLUGIN", "--json", *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_train_exact_resume_evaluate_and_export(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            code, output, error = self._cli(
                "train", str(CONFIG), "--output-dir", str(first)
            )
            self.assertEqual(code, 0, error)
            self.assertEqual(json.loads(output)["status"], "succeeded")
            step1 = first / "checkpoints" / "step-00000001"
            step2 = first / "checkpoints" / "step-00000002"
            self.assertTrue(step1.is_dir() and step2.is_dir())

            resumed = root / "resumed"
            code, output, error = self._cli(
                "train",
                str(CONFIG),
                "--output-dir",
                str(resumed),
                "--resume",
                str(step1),
                "--trusted-resume",
            )
            self.assertEqual(code, 0, error)
            resumed_step2 = resumed / "checkpoints" / "step-00000002"
            with (step2 / "state.pkl").open("rb") as handle:
                expected = pickle.load(handle)["objects"]["model"]
            with (resumed_step2 / "state.pkl").open("rb") as handle:
                actual = pickle.load(handle)["objects"]["model"]
            self.assertEqual(set(actual), set(expected))
            self.assertTrue(all(torch.equal(actual[key], expected[key]) for key in actual))

            evaluation = root / "evaluation"
            code, output, error = self._cli(
                "evaluate",
                str(CONFIG),
                "--output-dir",
                str(evaluation),
                "--checkpoint",
                str(step2),
                "--trusted-checkpoint",
                "--max-batches",
                "2",
            )
            self.assertEqual(code, 0, error)
            self.assertIn("loss/token_ce", json.loads(output)["metrics"])

            exported = root / "exported"
            code, output, error = self._cli(
                "export",
                str(CONFIG),
                "--checkpoint",
                str(step2),
                "--trusted-checkpoint",
                "--output-dir",
                str(exported),
                "--format",
                "torch",
            )
            self.assertEqual(code, 0, error)
            self.assertTrue((exported / "model.pt").is_file())
            self.assertTrue((exported / "export-manifest.json").is_file())

    def test_two_stage_pipeline_passes_physical_checkpoint_and_reuses_executor(self) -> None:
        base = load_run_spec(CONFIG).model_dump(mode="json")
        align = deepcopy(base["stage"])
        align["stage_id"] = "align"
        align["stage_type"] = "modality_alignment"
        align["optimization"]["max_steps"] = 1
        sft = deepcopy(base["stage"])
        sft["stage_id"] = "sft"
        sft["optimization"]["max_steps"] = 1
        pipeline = PipelineSpec(
            name="torch-toy-two-stage",
            seed=base["seed"],
            model=base["model"],
            stages=(align, sft),
            edges=(
                StageEdge(
                    from_stage="align",
                    to_stage="sft",
                    input_slot="model",
                    selector="last",
                ),
            ),
        )
        registry = ModelPluginRegistry()
        record = registry.load_explicit(f"{PLUGIN}:PLUGIN", allow_external=True)
        resolved, report = resolve_pipeline(
            pipeline,
            record.manifest,
            source=CONFIG,
        )
        self.assertTrue(report.valid, report.issues)
        assert resolved is not None

        with tempfile.TemporaryDirectory() as directory:
            executor = PipelineExecutor(
                resolved,
                plugin=record.plugin,
                output_dir=Path(directory),
            )
            first = executor.run()
            self.assertEqual(first.statuses, {"align": "succeeded", "sft": "succeeded"})
            align_checkpoint = first.outputs["align"]["checkpoint"]
            self.assertIsNotNone(align_checkpoint.uri)
            self.assertTrue(Path(align_checkpoint.uri).is_dir())

            # Reusing the same executor reconstructs its catalog from durable
            # state instead of failing duplicate artifact registration.
            resumed = executor.run(resume=True, trusted_resume=True)
            self.assertEqual(resumed.to_dict(), first.to_dict())
            sft_manifest = executor.catalog.resolve(resumed.outputs["sft"]["checkpoint"])
            self.assertEqual(sft_manifest.resume_level, "exact")
            self.assertEqual(sft_manifest.parents[0].artifact_id, align_checkpoint.artifact_id)
            self.assertEqual(sft_manifest.parents[0].uri, align_checkpoint.uri)


if __name__ == "__main__":
    unittest.main()
