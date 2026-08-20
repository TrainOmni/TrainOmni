from __future__ import annotations

import importlib.util
import io
import json
import pickle
import shutil
import sys
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.checkpoint import ScalarState
from trainomni.cli import main as cli_main
from trainomni.config import TrainingDiagnosticsSpec, load_run_spec, resolve_run
from trainomni.engines.peft import PeftError, apply_peft_if_requested
from trainomni.engines.torch_engine import (
    TorchEngineError,
    _capture_step_evidence,
    _finish_step_evidence,
)
from trainomni.models import ActivationCheckpointingReceipt
from trainomni.registry import ModelPluginRegistry
from trainomni.runtime import StageRunRequest, execute_stage

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PLUGIN = FRAMEWORK_ROOT / "tests" / "plugins" / "torch_toy_vlm_plugin.py"
CONFIG = FRAMEWORK_ROOT / "configs" / "examples" / "torch_toy_smoke.yaml"
FIXTURE = FRAMEWORK_ROOT / "tests" / "fixtures" / "datasets" / "torch_toy_sft.jsonl"


@unittest.skipUnless(TORCH_AVAILABLE, "requires optional torch runtime")
class FullParameterSFTContractTests(unittest.TestCase):
    def _cli(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main(["--plugin", f"{PLUGIN}:PLUGIN", "--json", *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def _p1_config(self) -> dict[str, Any]:
        value = load_run_spec(CONFIG).model_dump(mode="json")
        value["name"] = "torch_toy_full_parameter_p1"
        value["stage"]["data"]["datasets"][0]["uri"] = str(FIXTURE)
        for component_id, policy in value["stage"]["component_policy"].items():
            policy.update(
                {
                    "trainable": True,
                    "learning_rate": 0.1,
                    "weight_decay": 0.0,
                    "dtype": "bf16",
                }
            )
            if component_id in {"vision_encoder", "language_model"}:
                policy["activation_checkpointing"] = {
                    "use_reentrant": False,
                    "config": {"preserve_rng_state": True},
                }
        optimization = value["stage"]["optimization"]
        optimization.update(
            {
                "learning_rate": 0.1,
                "weight_decay": 0.0,
                "max_steps": 2,
                "optimizer_config": {
                    "implementation": "torch",
                    "foreach": False,
                    "kwargs": {},
                    "quantization": None,
                },
                "diagnostics": {
                    "record_gpu_memory": True,
                    "component_grad_norms": True,
                    "component_update_probes": True,
                    "require_finite_nonzero_gradients": True,
                    "require_parameter_updates": True,
                    "expected_trainable_numel": 1176,
                    "required_components": [
                        "vision_encoder",
                        "connector",
                        "language_model",
                    ],
                    "max_reserved_bytes": None,
                },
            }
        )
        value["stage"]["engine"].update(
            {"precision": "bf16", "config": {"device": "cpu"}}
        )
        value["stage"]["checkpoint"].update({"every_steps": 1})
        return value

    def _write_config(self, root: Path, value: Mapping[str, Any]) -> Path:
        path = root / "p1.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return path

    def assert_nested_equal(self, expected: Any, actual: Any) -> None:
        import torch

        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, Mapping):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual, strict=True):
                self.assert_nested_equal(expected_item, actual_item)
        else:
            self.assertEqual(expected, actual)

    def test_bf16_adamw_metadata_diagnostics_and_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root, self._p1_config())
            first = root / "first"
            code, output, error = self._cli(
                "train", str(config), "--output-dir", str(first)
            )
            self.assertEqual(code, 0, error or output)

            run_manifest = json.loads(
                (first / "run-manifest.json").read_text(encoding="utf-8")
            )
            metadata = run_manifest["metadata"]
            optimizer = metadata["optimizer"]
            self.assertEqual(optimizer["name"], "adamw")
            self.assertEqual(optimizer["implementation"], "torch")
            self.assertEqual(optimizer["package"]["name"], "torch")
            self.assertTrue(optimizer["package"]["version"])
            self.assertIs(optimizer["configured_kwargs"]["foreach"], False)
            self.assertIs(optimizer["actual_defaults"]["foreach"], False)
            self.assertIsNone(optimizer["quantization"])
            self.assertIn("torch.bfloat16", optimizer["state"]["state_dtypes"])
            self.assertEqual(optimizer["exact_resume"], "full_state_dict")

            self.assertEqual(metadata["trainable_numel"], 1176)
            self.assertEqual(
                set(metadata["activation_checkpointing"]),
                {"vision_encoder", "language_model"},
            )
            self.assertTrue(
                all(
                    not item["use_reentrant"]
                    for item in metadata["activation_checkpointing"].values()
                )
            )
            evidence = metadata["training_evidence"]
            self.assertEqual(
                evidence["schema_version"], "trainomni.training-evidence.v2"
            )
            for component_id in (
                "vision_encoder",
                "connector",
                "language_model",
            ):
                component = evidence["components"][component_id]
                self.assertGreater(component["grad_norm"], 0)
                self.assertTrue(component["grad_finite"])
                update = component["update_probe"]
                self.assertEqual(
                    update["method"], "exact_full_parameter_bitwise_scan"
                )
                self.assertTrue(update["exact"])
                self.assertGreater(update["probed_elements"], 1)
                self.assertGreater(update["changed_elements"], 0)
                self.assertGreater(update["changed_tensors"], 0)
                self.assertGreater(update["abs_update_l1"], 0)
            self.assertFalse(evidence["gpu_memory"]["available"])
            metric_records = [
                json.loads(line)
                for line in (first / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            metric_values = metric_records[-1]["values"]
            for component_id in (
                "vision_encoder",
                "connector",
                "language_model",
            ):
                prefix = f"components/{component_id}"
                self.assertGreater(
                    metric_values[f"{prefix}/changed_elements"], 0
                )
                self.assertGreater(
                    metric_values[f"{prefix}/changed_tensors"], 0
                )
                self.assertGreater(
                    metric_values[f"{prefix}/abs_update_l1"], 0
                )

            step1 = first / "checkpoints" / "step-00000001"
            step2 = first / "checkpoints" / "step-00000002"
            checkpoint_manifest = json.loads(
                (step2 / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint_manifest["metadata"]["optimizer"], optimizer
            )

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
                expected_state = pickle.load(handle)
            with (
                resumed / "checkpoints" / "step-00000002" / "state.pkl"
            ).open("rb") as handle:
                actual_state = pickle.load(handle)
            self.assert_nested_equal(expected_state, actual_state)
            resumed_manifest = json.loads(
                (
                    resumed
                    / "checkpoints"
                    / "step-00000002"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint_manifest["metadata"]["training_evidence"],
                resumed_manifest["metadata"]["training_evidence"],
            )

            tampered = root / "tampered-step-1"
            shutil.copytree(step1, tampered)
            tampered_manifest_path = tampered / "manifest.json"
            tampered_manifest = json.loads(
                tampered_manifest_path.read_text(encoding="utf-8")
            )
            tampered_manifest["metadata"]["optimizer"]["implementation"] = (
                "bitsandbytes"
            )
            tampered_manifest_path.write_text(
                json.dumps(tampered_manifest, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            code, output, error = self._cli(
                "train",
                str(config),
                "--output-dir",
                str(root / "tampered-resume"),
                "--resume",
                str(tampered),
                "--trusted-resume",
            )
            self.assertEqual(code, 2, error or output)
            self.assertIn("optimizer identity/state-dtype", output)

    def _run_bf16_update_probe(self, learning_rate: float) -> dict[str, Any]:
        import torch

        class TwoScaleComponent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor([1024.0, 1.0], dtype=torch.bfloat16)
                )

        model = TwoScaleComponent()
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [model.weight],
                    "component_id": "connector",
                }
            ],
            lr=learning_rate,
        )
        diagnostics = TrainingDiagnosticsSpec(
            record_gpu_memory=False,
            component_grad_norms=True,
            component_update_probes=True,
            update_probe_chunk_elements=1,
            require_finite_nonzero_gradients=True,
            require_parameter_updates=True,
            required_components=("connector",),
        )
        state = SimpleNamespace(
            context=SimpleNamespace(
                resolved=SimpleNamespace(
                    run=SimpleNamespace(
                        stage=SimpleNamespace(
                            optimization=SimpleNamespace(
                                diagnostics=diagnostics
                            )
                        )
                    )
                )
            ),
            torch=torch,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            step=ScalarState(),
            trainable_numel_by_component={"connector": 2},
        )
        model.weight.grad = torch.ones_like(model.weight)
        evidence, probes = _capture_step_evidence(state)
        optimizer.step()
        return _finish_step_evidence(state, evidence, probes)

    def test_exact_scan_accepts_bf16_partial_change_deterministically(self) -> None:
        first = self._run_bf16_update_probe(learning_rate=0.01)
        second = self._run_bf16_update_probe(learning_rate=0.01)
        self.assertEqual(first, second)
        update = first["components"]["connector"]["update_probe"]
        self.assertEqual(update["probed_elements"], 2)
        self.assertEqual(update["changed_elements"], 1)
        self.assertFalse(update["representative"]["bitwise_changed"])
        self.assertEqual(update["first_changed_element"]["flat_index"], 1)
        self.assertGreater(update["abs_update"], 0)
        self.assertGreater(update["abs_update_l1"], 0)

    def test_exact_scan_rejects_completely_unchanged_bf16_component(self) -> None:
        with self.assertRaisesRegex(
            TorchEngineError, "no exact numerical parameter-update evidence"
        ):
            self._run_bf16_update_probe(learning_rate=0.001)

    def test_bitsandbytes_request_never_falls_back_on_cpu(self) -> None:
        value = self._p1_config()
        value["stage"]["optimization"]["optimizer_config"] = {
            "implementation": "bitsandbytes",
            "foreach": None,
            "kwargs": {},
            "quantization": {
                "bits": 8,
                "min_8bit_size": 4096,
                "percentile_clipping": 100.0,
                "block_wise": True,
                "paged": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root, value)
            output_dir = root / "bnb"
            code, output, error = self._cli(
                "train", str(config), "--output-dir", str(output_dir)
            )
            self.assertEqual(code, 2, error or output)
            self.assertIn("requires a CUDA device", output)
            self.assertIn("no fallback", output.lower())
            self.assertFalse((output_dir / "checkpoints").exists())

    @unittest.skipUnless(
        importlib.util.find_spec("peft") is not None,
        "requires optional PEFT runtime",
    )
    def test_qlora_requires_plugin_quantization_and_invokes_kbit_prepare(self) -> None:
        value = self._p1_config()
        for policy in value["stage"]["component_policy"].values():
            policy.pop("activation_checkpointing", None)
        value["stage"]["component_policy"]["language_model"]["peft"] = {
            "method": "qlora",
            "rank": 2,
            "alpha": 4.0,
            "dropout": 0.0,
            "target_modules": ["language", "lm_head"],
            "modules_to_save": [],
            "task_type": "CAUSAL_LM",
            "config": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = load_run_spec(
                self._write_config(root, value)
            ).stage

            with self.assertRaisesRegex(PeftError, "4-bit or 8-bit"):
                apply_peft_if_requested(SimpleNamespace(), stage)

            quantized = SimpleNamespace(is_loaded_in_4bit=True)
            prepared = SimpleNamespace(name="prepared")
            wrapped = SimpleNamespace(name="wrapped")
            lora_config = SimpleNamespace(name="lora-config")
            with (
                patch(
                    "peft.prepare_model_for_kbit_training",
                    return_value=prepared,
                ) as prepare,
                patch("peft.LoraConfig", return_value=lora_config) as config,
                patch("peft.get_peft_model", return_value=wrapped) as wrap,
            ):
                result = apply_peft_if_requested(quantized, stage)
            self.assertIs(result, wrapped)
            prepare.assert_called_once_with(quantized)
            config.assert_called_once()
            self.assertEqual(config.call_args.kwargs["r"], 2)
            self.assertEqual(
                config.call_args.kwargs["target_modules"],
                ["language", "lm_head"],
            )
            wrap.assert_called_once_with(prepared, lora_config)

    def test_activation_checkpointing_requires_exact_plugin_receipts(self) -> None:
        registry = ModelPluginRegistry()
        base = registry.load_explicit(
            f"{PLUGIN}:PLUGIN", allow_external=True
        ).plugin

        class PluginWithoutCheckpointHook:
            manifest = base.manifest
            configure_activation_checkpointing = None

            def __getattr__(self, name: str) -> Any:
                return getattr(base, name)

        class PluginWithWrongReceipt:
            manifest = base.manifest

            def __getattr__(self, name: str) -> Any:
                return getattr(base, name)

            def configure_activation_checkpointing(self, bundle, requests):
                return {
                    component_id: ActivationCheckpointingReceipt(
                        component_id=component_id,
                        implementation="wrong-reentrant-test",
                        use_reentrant=not request.use_reentrant,
                    )
                    for component_id, request in requests.items()
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root, self._p1_config())
            run = load_run_spec(config)
            plugin = PluginWithoutCheckpointHook()
            resolved, report = resolve_run(run, plugin.manifest, source=config)
            self.assertTrue(report.valid, report.issues)
            assert resolved is not None
            with self.assertRaisesRegex(
                RuntimeError, "requires model plugin method"
            ):
                execute_stage(
                    StageRunRequest(
                        resolved=resolved,
                        plugin=plugin,
                        output_dir=root / "no-hook",
                        input_artifacts={},
                    )
                )
            wrong = PluginWithWrongReceipt()
            wrong_resolved, report = resolve_run(
                run, wrong.manifest, source=config
            )
            self.assertTrue(report.valid, report.issues)
            assert wrong_resolved is not None
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                execute_stage(
                    StageRunRequest(
                        resolved=wrong_resolved,
                        plugin=wrong,
                        output_dir=root / "wrong-receipt",
                        input_artifacts={},
                    )
                )


if __name__ == "__main__":
    unittest.main()
