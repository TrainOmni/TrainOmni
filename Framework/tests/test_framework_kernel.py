from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pydantic import ValidationError

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.cli import main as cli_main
from trainomni.config import (
    ConfigLoadError,
    RunSpec,
    canonical_fingerprint,
    load_run_spec,
    resolve_run,
)
from trainomni.contracts import ArtifactManifest, ArtifactRef
from trainomni.engines import (
    ENGINE_API_VERSION,
    EngineCapabilities,
    EngineKind,
    EngineManifest,
    EngineRequirements,
    negotiate_engine,
)
from trainomni.models import (
    MODEL_PLUGIN_API_VERSION,
    ModelCapabilities,
    ModelPluginManifest,
    validate_plugin_components,
)
from trainomni.objectives import (
    LossOutput,
    LossTerm,
    ObjectiveManifest,
    ObjectiveRequirements,
)
from trainomni.registry import ModelPluginRegistry, PluginRegistryError

TOY_PLUGIN = FRAMEWORK_ROOT / "tests" / "plugins" / "toy_vlm_plugin.py"
TOY_CONFIG = FRAMEWORK_ROOT / "configs" / "examples" / "toy_alignment.yaml"
TOY_SPEC = f"{TOY_PLUGIN}:PLUGIN"


class FrameworkKernelTests(unittest.TestCase):
    def test_objective_contract_requires_named_normalized_loss_terms(self) -> None:
        manifest = ObjectiveManifest(
            objective_id="masked-causal-lm",
            objective_version="1.0.0",
            requirements=ObjectiveRequirements(
                sample_objectives=frozenset({"cpt", "sft"})
            ),
            supported_engines=frozenset({"torch", "nemo"}),
        )
        output = LossOutput(
            total=1.0,
            terms={"token_ce": LossTerm(value=1.0, denominator=16)},
            metrics={"token_accuracy": 0.5},
            counts={"loss_tokens": 16},
        )
        self.assertEqual(manifest.objective_id, "masked-causal-lm")
        self.assertEqual(output.terms["token_ce"].denominator, 16)
        with self.assertRaisesRegex(ValueError, "denominator"):
            LossTerm(value=0.0, denominator=0)

    def test_loop_and_delegated_engine_contracts_negotiate_all_fields(self) -> None:
        capabilities = EngineCapabilities(
            stage_types=frozenset({"instruction_sft"}),
            objectives=frozenset({"sft"}),
            parallelism=frozenset({"single", "ddp"}),
            precisions=frozenset({"fp32", "bf16"}),
            resume_levels=frozenset({"exact"}),
        )
        manifest = EngineManifest(
            engine_id="torch",
            engine_version="0.1.0",
            kind=EngineKind.LOOP,
            capabilities=capabilities,
        )
        self.assertEqual(manifest.api_version, ENGINE_API_VERSION)
        report = negotiate_engine(
            EngineRequirements(
                stage_type="online_rl",
                objective="grpo",
                parallelism="fsdp2",
                precision="fp8",
                resume_level="stage_boundary",
                require_generation=True,
                require_multiple_models=True,
                require_rollout=True,
            ),
            capabilities,
        )
        self.assertFalse(report.valid)
        self.assertEqual(
            {issue.code for issue in report.errors},
            {
                "engine.stage_type",
                "engine.objective",
                "engine.parallelism",
                "engine.precision",
                "engine.resume_level",
                "engine.generation",
                "engine.multiple_models",
                "engine.rollout",
            },
        )

    def test_artifact_contract_validates_resume_level(self) -> None:
        parent = ArtifactRef("base-model", "best")
        manifest = ArtifactManifest(
            artifact_id="aligned-model",
            artifact_type="checkpoint",
            run_id="run-1",
            stage_id="align-1",
            fingerprint="a" * 64,
            resume_level="exact",
            parents=(parent,),
        )
        self.assertEqual(str(parent), "artifact://base-model/best")
        self.assertEqual(manifest.parents, (parent,))
        with self.assertRaisesRegex(ValueError, "resume_level"):
            ArtifactManifest(
                artifact_id="bad",
                artifact_type="checkpoint",
                run_id="run-1",
                stage_id="align-1",
                fingerprint="b" * 64,
                resume_level="implicit",
            )

    def test_manifest_rejects_invalid_identity_and_api(self) -> None:
        capabilities = ModelCapabilities(
            modalities=frozenset({"text"}),
            content_blocks=frozenset({"text"}),
            objectives=frozenset({"sft"}),
        )
        with self.assertRaisesRegex(ValueError, "plugin_id"):
            ModelPluginManifest(
                "Bad Plugin", "1", capabilities, component_ids=("language_model",)
            )
        with self.assertRaisesRegex(ValueError, "unsupported model plugin API"):
            ModelPluginManifest(
                "good-plugin",
                "1",
                capabilities,
                component_ids=("language_model",),
                api_version="future.v9",
            )
        self.assertEqual(MODEL_PLUGIN_API_VERSION, "trainomni.model-plugin.v1")

    def test_explicit_plugin_requires_trust_and_core_zero_edit(self) -> None:
        registry = ModelPluginRegistry()
        with self.assertRaisesRegex(PluginRegistryError, "allow_external"):
            registry.load_explicit(TOY_SPEC, allow_external=False)
        record = registry.load_explicit(TOY_SPEC, allow_external=True)
        self.assertEqual(record.manifest.plugin_id, "toy-vlm")
        self.assertTrue(record.external)
        self.assertEqual(registry.get("toy-vlm"), record)

        bundle = record.plugin.build({})
        report = validate_plugin_components(
            record.plugin, bundle, bundle.parameter_names
        )
        self.assertTrue(report.valid, report.issues)

    def test_registry_rejects_conflicting_plugin_id(self) -> None:
        registry = ModelPluginRegistry()
        record = registry.load_explicit(TOY_SPEC, allow_external=True)

        class Conflicting:
            manifest = ModelPluginManifest(
                plugin_id="toy-vlm",
                plugin_version="2.0.0",
                capabilities=record.manifest.capabilities,
                component_ids=record.manifest.component_ids,
            )

            capabilities = record.plugin.capabilities
            build = record.plugin.build
            component_catalog = record.plugin.component_catalog
            validate_sample = record.plugin.validate_sample
            encode = record.plugin.encode
            collate = record.plugin.collate
            export = record.plugin.export

        with self.assertRaisesRegex(PluginRegistryError, "already registered"):
            registry.register(Conflicting(), source="conflict", external=True)

    def test_yaml_config_is_strict_and_fingerprint_is_stable(self) -> None:
        spec = load_run_spec(TOY_CONFIG)
        self.assertEqual(spec.model.plugin, "toy-vlm")
        self.assertEqual(spec.stage.stage_type, "modality_alignment")
        first = canonical_fingerprint(spec)
        second = canonical_fingerprint(spec)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(
            canonical_fingerprint({"values": frozenset({"image", "text"})}),
            canonical_fingerprint({"values": frozenset({"text", "image"})}),
        )

        value = spec.model_dump(mode="json")
        value["unexpected"] = True
        with self.assertRaises(ValidationError):
            RunSpec.model_validate(value)

    def test_optimizer_contract_rejects_ambiguous_or_implicit_quantization(self) -> None:
        base = load_run_spec(TOY_CONFIG).model_dump(mode="json")

        unknown = json.loads(json.dumps(base))
        unknown["stage"]["optimization"]["optimizer_config"] = {
            "implementation": "torch",
            "foreach": False,
            "kwargs": {},
            "quantization": None,
            "fallback": "adamw",
        }
        with self.assertRaisesRegex(ValidationError, "fallback"):
            RunSpec.model_validate(unknown)

        ambiguous = json.loads(json.dumps(base))
        ambiguous["stage"]["optimization"]["optimizer_config"] = {
            "implementation": "torch",
            "foreach": False,
            "kwargs": {},
            "quantization": None,
        }
        ambiguous["stage"]["optimization"]["config"]["optimizer"] = {
            "foreach": False
        }
        with self.assertRaisesRegex(ValidationError, "do not combine"):
            RunSpec.model_validate(ambiguous)

        implicit_quantization = json.loads(json.dumps(base))
        implicit_quantization["stage"]["optimization"]["optimizer_config"] = {
            "implementation": "bitsandbytes",
            "foreach": None,
            "kwargs": {},
            "quantization": None,
        }
        with self.assertRaisesRegex(ValidationError, "explicit quantization"):
            RunSpec.model_validate(implicit_quantization)

    def test_fingerprint_is_stable_across_python_hash_seeds(self) -> None:
        code = (
            "from trainomni.config import canonical_fingerprint, load_run_spec; "
            f"print(canonical_fingerprint(load_run_spec({str(TOY_CONFIG)!r})))"
        )
        fingerprints = []
        for seed in ("1", "987654"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            environment["PYTHONPATH"] = str(SRC_ROOT)
            fingerprints.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    cwd=FRAMEWORK_ROOT,
                    env=environment,
                    text=True,
                ).strip()
            )
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_config_loader_reports_precise_unknown_field(self) -> None:
        value = load_run_spec(TOY_CONFIG).model_dump(mode="json")
        value["stage"]["engien"] = {"backend": "torch"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ConfigLoadError) as raised:
                load_run_spec(path)
        self.assertIn("stage.engien", str(raised.exception))
        self.assertIn("extra_forbidden", str(raised.exception))

    def test_resolve_rejects_capability_and_component_mismatch(self) -> None:
        spec_value = load_run_spec(TOY_CONFIG).model_dump(mode="json")
        spec_value["stage"]["data"]["packing"] = True
        spec_value["stage"]["data"]["padding_free"] = True
        spec_value["stage"]["component_policy"]["audio_encoder"] = {
            "trainable": True
        }
        spec = RunSpec.model_validate(spec_value)
        registry = ModelPluginRegistry()
        record = registry.load_explicit(TOY_SPEC, allow_external=True)
        resolved, report = resolve_run(spec, record.manifest)
        self.assertIsNone(resolved)
        self.assertEqual(
            {issue.code for issue in report.errors},
            {"capability.padding_free", "plugin.component_policy"},
        )

    def test_cli_validate_inspect_and_dry_run(self) -> None:
        commands = (
            ["--plugin", TOY_SPEC, "--json", "validate", str(TOY_CONFIG)],
            [
                "--plugin",
                TOY_SPEC,
                "--json",
                "inspect",
                "model",
                str(TOY_CONFIG),
            ],
            ["--plugin", TOY_SPEC, "--json", "dry-run", str(TOY_CONFIG)],
        )
        payloads = []
        for command in commands:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(command)
            self.assertEqual(exit_code, 0, stdout.getvalue())
            payloads.append(json.loads(stdout.getvalue()))
        self.assertTrue(payloads[0]["valid"])
        self.assertEqual(payloads[1]["plugin"]["plugin_id"], "toy-vlm")
        self.assertFalse(payloads[2]["plan"]["will_load_weights"])
        self.assertFalse(payloads[2]["plan"]["will_execute_training"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli_main(
                ["--plugin", TOY_SPEC, "--json", "plugins", "list"]
            )
        self.assertEqual(exit_code, 0)
        plugins_payload = json.loads(stdout.getvalue())
        self.assertEqual(plugins_payload["loaded"][0]["plugin_id"], "toy-vlm")

    def test_cli_refuses_untrusted_recipe_plugin(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = cli_main(["validate", str(TOY_CONFIG)])
        self.assertEqual(exit_code, 2)
        self.assertIn("is not loaded", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
