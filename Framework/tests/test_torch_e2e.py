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
from typing import Any

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.cli import main as cli_main
from trainomni.config import load_run_spec, resolve_run
from trainomni.models import ModelBundle
from trainomni.recipes import PipelineSpec, StageEdge, resolve_pipeline
from trainomni.registry import ModelPluginRegistry
from trainomni.runtime import PipelineExecutor, evaluate_run

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

    def _write_variant(
        self, root: Path, name: str, mutate: Any
    ) -> Path:
        value = load_run_spec(CONFIG).model_dump(mode="json")
        value["name"] = name
        value["stage"]["data"]["datasets"][0]["uri"] = str(
            FRAMEWORK_ROOT / "tests" / "fixtures" / "datasets" / "torch_toy_sft.jsonl"
        )
        mutate(value)
        target = root / f"{name}.json"
        target.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return target

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
            evaluation_payload = json.loads(output)
            self.assertIn("loss/token_ce", evaluation_payload["metrics"])
            self.assertEqual(
                evaluation_payload["execution"],
                {
                    "backend": "torch",
                    "device": "cpu",
                    "mode": "local",
                    "precision": "fp32",
                },
            )

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

    def test_gradient_accumulation_exact_resume_uses_microstep_cursor(self) -> None:
        import torch

        def mutate(value: dict[str, Any]) -> None:
            stage = value["stage"]
            stage["optimization"]["max_steps"] = 2
            stage["optimization"]["gradient_accumulation_steps"] = 2
            stage["checkpoint"]["every_steps"] = 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_variant(
                root, "torch_toy_accumulation_exact", mutate
            )
            first = root / "first"
            code, output, error = self._cli(
                "train", str(config), "--output-dir", str(first)
            )
            self.assertEqual(code, 0, error or output)
            step1 = first / "checkpoints" / "step-00000001"
            step2 = first / "checkpoints" / "step-00000002"
            with (step2 / "state.pkl").open("rb") as handle:
                expected = pickle.load(handle)["objects"]
            self.assertEqual(expected["step"]["value"], 2)
            self.assertEqual(expected["microstep"]["value"], 4)

            resumed = root / "resumed"
            code, output, error = self._cli(
                "train",
                str(config),
                "--output-dir",
                str(resumed),
                "--resume",
                str(step1),
                "--trusted-resume",
            )
            self.assertEqual(code, 0, error or output)
            with (
                resumed / "checkpoints" / "step-00000002" / "state.pkl"
            ).open("rb") as handle:
                actual = pickle.load(handle)["objects"]
            self.assertEqual(actual["microstep"]["value"], 4)
            self.assertEqual(set(actual["model"]), set(expected["model"]))
            self.assertTrue(
                all(
                    torch.equal(actual["model"][key], expected["model"][key])
                    for key in actual["model"]
                )
            )
            self.assertEqual(actual["data"], expected["data"])

    def test_torch_compile_eager_backend_trains_and_checkpoints(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            stage = value["stage"]
            stage["optimization"]["max_steps"] = 1
            stage["engine"]["config"] = {
                "device": "cpu",
                "compile": {"backend": "eager"},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_variant(root, "torch_toy_compile_eager", mutate)
            output_dir = root / "compiled"
            code, output, error = self._cli(
                "train", str(config), "--output-dir", str(output_dir)
            )
            self.assertEqual(code, 0, error or output)
            self.assertTrue(
                (output_dir / "checkpoints" / "step-00000001" / "manifest.json").is_file()
            )

    @unittest.skipUnless(
        importlib.util.find_spec("peft") is not None,
        "requires optional PEFT runtime",
    )
    def test_lora_trains_only_adapters_and_exact_resumes(self) -> None:
        import torch

        def mutate(value: dict[str, Any]) -> None:
            stage = value["stage"]
            stage["optimization"]["max_steps"] = 2
            stage["checkpoint"]["every_steps"] = 1
            for component_id, policy in stage["component_policy"].items():
                policy["trainable"] = component_id == "language_model"
                policy.pop("learning_rate", None)
                policy.pop("gradient_clip", None)
            language = stage["component_policy"]["language_model"]
            language["learning_rate"] = 0.01
            language["peft"] = {
                "method": "lora",
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target_modules": ["language", "lm_head"],
                "modules_to_save": [],
                "task_type": "FEATURE_EXTRACTION",
                "config": {},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_variant(root, "torch_toy_lora_exact", mutate)
            first = root / "first"
            code, output, error = self._cli(
                "train", str(config), "--output-dir", str(first)
            )
            self.assertEqual(code, 0, error or output)
            manifest = json.loads(
                (first / "run-manifest.json").read_text(encoding="utf-8")
            )
            trainable = manifest["metadata"]["trainable_numel"]
            self.assertGreater(trainable, 0)
            self.assertLess(trainable, 1088)

            step1 = first / "checkpoints" / "step-00000001"
            step2 = first / "checkpoints" / "step-00000002"
            resumed = root / "resumed"
            code, output, error = self._cli(
                "train",
                str(config),
                "--output-dir",
                str(resumed),
                "--resume",
                str(step1),
                "--trusted-resume",
            )
            self.assertEqual(code, 0, error or output)
            with (step2 / "state.pkl").open("rb") as handle:
                expected = pickle.load(handle)["objects"]["model"]
            with (
                resumed / "checkpoints" / "step-00000002" / "state.pkl"
            ).open("rb") as handle:
                actual = pickle.load(handle)["objects"]["model"]
            self.assertEqual(set(actual), set(expected))
            self.assertTrue(
                all(torch.equal(actual[key], expected[key]) for key in actual)
            )

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

    def test_evaluate_uses_configured_device_precision_and_inference_mode(self) -> None:
        import torch

        registry = ModelPluginRegistry()
        base = registry.load_explicit(
            f"{PLUGIN}:PLUGIN", allow_external=True
        ).plugin

        class TrackingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(()))
                self.to_calls: list[str] = []
                self.eval_calls = 0
                self.observations: list[dict[str, object]] = []

            def to(self, *args, **kwargs):
                result = super().to(*args, **kwargs)
                self.to_calls.append(str(self.anchor.device))
                return result

            def eval(self):
                self.eval_calls += 1
                return super().eval()

            def forward(self, input_ids, labels, **kwargs):
                try:
                    autocast_enabled = torch.is_autocast_enabled("cpu")
                except TypeError:  # pragma: no cover - older torch compatibility
                    autocast_enabled = torch.is_autocast_cpu_enabled()
                self.observations.append(
                    {
                        "training": self.training,
                        "inference": torch.is_inference_mode_enabled(),
                        "autocast": autocast_enabled,
                        "input_device": input_ids.device.type,
                        "parameter_device": self.anchor.device.type,
                    }
                )
                loss = self.anchor * 0 + input_ids.float().sum() * 0 + 2.5
                return {"loss": loss}

        class TrackingPlugin:
            manifest = base.manifest

            def __init__(self) -> None:
                self.primary: TrackingModel | None = None
                self.teacher: TrackingModel | None = None

            def __getattr__(self, name: str):
                return getattr(base, name)

            def build(self, context):
                self.primary = TrackingModel()
                self.teacher = TrackingModel()
                return ModelBundle(
                    self.primary,
                    auxiliary_models={"teacher": self.teacher},
                )

        plugin = TrackingPlugin()
        run = load_run_spec(CONFIG)
        engine = run.stage.engine.model_copy(
            update={"precision": "bf16", "config": {"device": "cpu"}}
        )
        stage = run.stage.model_copy(update={"engine": engine})
        run = run.model_copy(update={"stage": stage})
        resolved, report = resolve_run(run, plugin.manifest, source=CONFIG)
        self.assertTrue(report.valid, report.issues)
        assert resolved is not None

        with tempfile.TemporaryDirectory() as directory:
            payload = evaluate_run(
                resolved,
                plugin,
                output_dir=Path(directory),
                max_batches=1,
            )

        self.assertEqual(payload["metrics"]["loss/token_ce"], 2.5)
        self.assertEqual(
            payload["execution"],
            {
                "backend": "torch",
                "device": "cpu",
                "mode": "local",
                "precision": "bf16",
            },
        )
        assert plugin.primary is not None and plugin.teacher is not None
        self.assertEqual(plugin.primary.to_calls, ["cpu"])
        self.assertEqual(plugin.teacher.to_calls, ["cpu"])
        self.assertGreaterEqual(plugin.primary.eval_calls, 1)
        self.assertGreaterEqual(plugin.teacher.eval_calls, 1)
        self.assertEqual(
            plugin.primary.observations,
            [
                {
                    "training": False,
                    "inference": True,
                    "autocast": True,
                    "input_device": "cpu",
                    "parameter_device": "cpu",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
