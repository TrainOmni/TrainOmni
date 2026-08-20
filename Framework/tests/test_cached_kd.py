from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pickle
import shutil
import sys
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.checkpoint import LocalCheckpointManager, ObjectState, StateRegistry
from trainomni.config import canonical_fingerprint, load_run_spec, resolve_run
from trainomni.contracts import (
    ArtifactRef,
    BatchBudget,
    BatchItem,
    BatchPlan,
    CostVector,
)
from trainomni.models import ModelBatch
from trainomni.objectives import (
    AssetFileIdentity,
    AssetSetIdentity,
    CachedLogitKDError,
    CacheSampleIdentity,
    CheckpointIdentity,
    DataIdentity,
    LogitCacheIdentity,
    LogitTensorIdentity,
    LossPositionIdentity,
    ModelIdentity,
    OfflineDenseLogitCacheManifest,
    OfflineDenseLogitKDObjective,
    asset_set_digest,
    data_identity_digest,
    integer_tensor_digest,
    loss_position_digest,
)
from trainomni.registry import ModelPluginRegistry
from trainomni.runtime import StageRunRequest, execute_stage

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PLUGIN_PATH = FRAMEWORK_ROOT / "tests" / "plugins" / "torch_toy_vlm_plugin.py"
BASE_CONFIG = FRAMEWORK_ROOT / "configs" / "examples" / "torch_toy_smoke.yaml"
DATASET = FRAMEWORK_ROOT / "tests" / "fixtures" / "datasets" / "torch_toy_sft.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class _KDFixture:
    config_path: Path
    plugin: Any
    student_checkpoint: Path
    teacher_checkpoint: Path
    cache_manifest: Path
    cache_tensor: Path
    artifact: ArtifactRef


@unittest.skipUnless(TORCH_AVAILABLE, "requires optional torch runtime")
class OfflineDenseLogitKDTests(unittest.TestCase):
    def _plugin(self) -> Any:
        return ModelPluginRegistry().load_explicit(
            f"{PLUGIN_PATH}:PLUGIN", allow_external=True
        ).plugin

    def _checkpoint(
        self, root: Path, plugin: Any, name: str, run_fingerprint: str
    ) -> Path:
        import torch

        torch.manual_seed(1234)
        bundle = plugin.build({})
        registry = StateRegistry()
        registry.register("model", ObjectState(bundle.model))
        return LocalCheckpointManager(root).save(
            name,
            registry,
            metadata={"run_fingerprint": run_fingerprint},
        )

    def _fixture(self, root: Path) -> _KDFixture:
        import torch

        plugin = self._plugin()
        student_run = "1" * 64
        teacher_run = "2" * 64
        student = self._checkpoint(
            root / "student", plugin, "step-00000000", student_run
        )
        teacher = self._checkpoint(
            root / "teacher", plugin, "step-00000002", teacher_run
        )
        artifact = ArtifactRef(
            "student", selector="step-00000000", uri=str(student)
        )

        dataset_path = root / "kd-dataset.jsonl"
        first_record = DATASET.read_text(encoding="utf-8").splitlines()[0]
        dataset_path.write_text(first_record + "\n", encoding="utf-8")

        cache_root = root / "cache"
        cache_root.mkdir()
        input_ids = torch.tensor([5, 2, 4, 3, 4], dtype=torch.long)
        labels = torch.tensor([-100, -100, -100, 3, 4], dtype=torch.long)
        positions = (3, 4)
        targets = (3, 4)
        logits = (
            torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64) / 31
        ).to(torch.bfloat16)
        payload = bytes(logits.view(torch.uint8).reshape(-1).tolist())
        tensor_path = cache_root / "torch-toy-001.bf16"
        tensor_path.write_bytes(payload)

        source_sha = _sha256(dataset_path)
        asset_file = AssetFileIdentity(
            path=str(dataset_path),
            sha256=source_sha,
            size_bytes=dataset_path.stat().st_size,
        )

        def assets(kind: str) -> AssetSetIdentity:
            metadata = {"kind": kind, "revision": "toy-pinned"}
            return AssetSetIdentity(
                identity_sha256=asset_set_digest((asset_file,), metadata),
                files=(asset_file,),
                metadata=metadata,
            )

        sample = CacheSampleIdentity(
            sample_id="torch-toy-001",
            source_index=0,
            canonical_sample_sha256=hashlib.sha256(
                b"torch-toy-001-canonical"
            ).hexdigest(),
            input_ids_sha256=integer_tensor_digest(input_ids),
            labels_sha256=integer_tensor_digest(labels),
            assistant_positions=positions,
            target_token_ids=targets,
            teacher_logits=LogitTensorIdentity(
                file=tensor_path.name,
                shape=(2, 64),
                dtype="bfloat16",
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
        )
        data_base = DataIdentity(
            source_path=str(dataset_path),
            source_sha256=source_sha,
            split_fingerprint="3" * 64,
            identity_sha256="0" * 64,
            metadata={"reader": "jsonl", "seed": 20260820},
        )
        data = data_base.model_copy(
            update={"identity_sha256": data_identity_digest(data_base, (sample,))}
        )
        loss_positions = LossPositionIdentity(
            identity_sha256=loss_position_digest((sample,)),
            total_positions=2,
        )
        student_identity = CheckpointIdentity(
            artifact=str(artifact),
            path=str(student),
            state_sha256=_sha256(student / "state.pkl"),
            manifest_sha256=_sha256(student / "manifest.json"),
            run_fingerprint=student_run,
        )
        teacher_identity = CheckpointIdentity(
            artifact="artifact://teacher/step-00000002",
            path=str(teacher),
            state_sha256=_sha256(teacher / "state.pkl"),
            manifest_sha256=_sha256(teacher / "manifest.json"),
            run_fingerprint=teacher_run,
        )
        model_assets = assets("model")
        tokenizer = assets("tokenizer")
        processor = assets("processor")
        content_sha = hashlib.sha256(payload).hexdigest()
        manifest = OfflineDenseLogitCacheManifest(
            cache_id="torch-toy-dense-kd-v1",
            producer_code_revision="test-revision",
            teacher=teacher_identity,
            student=student_identity,
            model=ModelIdentity(
                plugin_id=plugin.manifest.plugin_id,
                plugin_version=plugin.manifest.plugin_version,
                assets=model_assets,
            ),
            tokenizer=tokenizer,
            processor=processor,
            data=data,
            loss_positions=loss_positions,
            logits=LogitCacheIdentity(
                vocab_size=64,
                total_positions=2,
                total_bytes=len(payload),
                content_sha256=content_sha,
            ),
            samples=(sample,),
        )
        manifest_path = cache_root / "manifest.json"
        manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

        config = load_run_spec(BASE_CONFIG).model_dump(mode="json")
        config["name"] = "torch_toy_offline_dense_logit_kd"
        stage = config["stage"]
        stage.update(
            {
                "stage_id": "offline_dense_logit_kd",
                "stage_type": "reasoning_distillation",
                "objective": "sft",
                "objective_impl": "offline-dense-logit-kd",
                "inputs": {"model": str(artifact)},
            }
        )
        stage["data"]["datasets"][0]["uri"] = str(dataset_path)
        stage["component_policy"] = {
            "vision_encoder": {"trainable": False, "dtype": "bf16"},
            "connector": {
                "trainable": True,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dtype": "fp32",
                "gradient_clip": 1.0,
            },
            "language_model": {"trainable": False, "dtype": "bf16"},
        }
        stage["optimization"].update(
            {
                "optimizer": "adamw",
                "optimizer_config": {
                    "implementation": "torch",
                    "foreach": False,
                    "kwargs": {},
                    "quantization": None,
                },
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "max_steps": 2,
                "diagnostics": {
                    "record_gpu_memory": False,
                    "component_grad_norms": True,
                    "component_update_probes": True,
                    "update_probe_chunk_elements": 32,
                    "require_finite_nonzero_gradients": True,
                    "require_parameter_updates": True,
                    "expected_trainable_numel": 72,
                    "required_components": ["connector"],
                    "max_reserved_bytes": None,
                },
            }
        )
        stage["engine"].update(
            {"precision": "bf16", "config": {"device": "cpu"}}
        )
        stage["checkpoint"].update({"every_steps": 1})
        config["metadata"] = {
            "offline_dense_logit_kd": {
                "cache_manifest": str(manifest_path),
                "cache_manifest_sha256": _sha256(manifest_path),
                "cache_content_sha256": content_sha,
                "teacher_state_sha256": teacher_identity.state_sha256,
                "teacher_manifest_sha256": teacher_identity.manifest_sha256,
                "teacher_run_fingerprint": teacher_identity.run_fingerprint,
                "student_state_sha256": student_identity.state_sha256,
                "student_manifest_sha256": student_identity.manifest_sha256,
                "student_run_fingerprint": student_identity.run_fingerprint,
                "model_identity_sha256": model_assets.identity_sha256,
                "tokenizer_sha256": tokenizer.identity_sha256,
                "processor_sha256": processor.identity_sha256,
                "data_sha256": data.identity_sha256,
                "loss_positions_sha256": loss_positions.identity_sha256,
                "temperature": 2.0,
                "ce_weight": 0.5,
                "kd_weight": 0.5,
                "vocab_size": 64,
                "cache_dtype": "bfloat16",
                "max_cache_bytes": 1024 * 1024,
            }
        }
        config_path = root / "kd.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return _KDFixture(
            config_path=config_path,
            plugin=plugin,
            student_checkpoint=student,
            teacher_checkpoint=teacher,
            cache_manifest=manifest_path,
            cache_tensor=tensor_path,
            artifact=artifact,
        )

    def _execute(
        self,
        fixture: _KDFixture,
        output_dir: Path,
        *,
        resume: Path | None = None,
    ) -> Any:
        spec = load_run_spec(fixture.config_path)
        resolved, report = resolve_run(
            spec, fixture.plugin.manifest, source=fixture.config_path
        )
        self.assertTrue(report.valid, report.issues)
        assert resolved is not None
        return execute_stage(
            StageRunRequest(
                resolved=resolved,
                plugin=fixture.plugin,
                output_dir=output_dir,
                input_artifacts={"model": fixture.artifact},
                resume_from=str(resume) if resume is not None else None,
                trusted_resume=resume is not None,
                trusted_input_artifacts=True,
            )
        )

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
            self.assertEqual(type(expected), type(actual))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual, strict=True):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_fp32_ce_teacher_kl_weighting_and_frozen_gradient_path(self) -> None:
        import torch

        class Student(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.connector = torch.nn.Linear(2, 2, bias=False)
                self.language = torch.nn.Linear(2, 3, bias=False)
                for parameter in self.language.parameters():
                    parameter.requires_grad_(False)

            def forward(self, features, labels):
                return {"logits": self.language(self.connector(features))}

        torch.manual_seed(7)
        model = Student()
        features = torch.tensor(
            [[[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75]]]
        )
        positions = torch.tensor([[1, 2]], dtype=torch.long)
        targets = torch.tensor([[1, 2]], dtype=torch.long)
        teacher = torch.tensor(
            [[[0.25, -0.5, 1.0], [1.5, 0.0, -0.25]]],
            dtype=torch.bfloat16,
        )
        plan = BatchPlan(
            items=(BatchItem("sample", 0, CostVector(text_tokens=3)),),
            total_cost=CostVector(text_tokens=3),
            budget=BatchBudget(max_samples=1, max_text_tokens=3),
        )
        batch = ModelBatch(
            sample_ids=("sample",),
            model_inputs={
                "features": features,
                "labels": torch.tensor([[-100, 1, 2]]),
                "kd_teacher_logits": teacher,
                "kd_assistant_positions": positions,
                "kd_position_mask": torch.ones_like(positions, dtype=torch.bool),
                "kd_target_token_ids": targets,
                "kd_cache_identity": "a" * 64,
            },
            plan=plan,
            trace={
                "offline_dense_logit_kd": {
                    "cache_content_sha256": "a" * 64,
                    "loss": {
                        "temperature": 2.0,
                        "ce_weight": 0.5,
                        "kd_weight": 0.5,
                    },
                }
            },
        )
        with self.assertRaisesRegex(CachedLogitKDError, "forbids a live teacher"):
            OfflineDenseLogitKDObjective().compute(
                {"model": model, "teacher": object()}, batch
            )
        output = OfflineDenseLogitKDObjective().compute({"model": model}, batch)
        logits = model(features=features, labels=batch.model_inputs["labels"])[
            "logits"
        ]
        selected = logits[0, :2].float()
        expected_ce = torch.nn.functional.cross_entropy(selected, targets[0])
        teacher_log = torch.nn.functional.log_softmax(teacher[0].float() / 2, -1)
        student_log = torch.nn.functional.log_softmax(selected / 2, -1)
        expected_kl = (
            teacher_log.exp() * (teacher_log - student_log)
        ).sum(-1).mean() * 4
        expected_total = 0.5 * expected_ce + 0.5 * expected_kl
        self.assertEqual(output.total.dtype, torch.float32)
        self.assertTrue(torch.allclose(output.terms["token_ce"].value, expected_ce))
        self.assertTrue(
            torch.allclose(output.terms["teacher_kl"].value, expected_kl)
        )
        self.assertTrue(torch.allclose(output.total, expected_total))
        self.assertEqual(output.counts["loss_tokens"], 2)
        output.total.backward()
        self.assertGreater(model.connector.weight.grad.norm().item(), 0)
        self.assertIsNone(model.language.weight.grad)

    def test_cache_identity_e2e_metrics_checkpoint_and_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            first = root / "first"
            result = self._execute(fixture, first)
            self.assertEqual(result.status, "succeeded")
            for name in (
                "token_ce",
                "teacher_kl",
                "weighted_ce",
                "weighted_teacher_kl",
                "total",
            ):
                self.assertTrue(math.isfinite(result.metrics[name]))
            step1 = first / "checkpoints" / "step-00000001"
            step2 = first / "checkpoints" / "step-00000002"
            manifest = json.loads(
                (step2 / "manifest.json").read_text(encoding="utf-8")
            )
            objective = manifest["metadata"]["objective"]
            self.assertEqual(objective["objective_id"], "offline-dense-logit-kd")
            evidence = manifest["metadata"]["objective_evidence"]
            for name in (
                "token_ce",
                "teacher_kl",
                "weighted_ce",
                "weighted_teacher_kl",
                "total",
            ):
                self.assertTrue(math.isfinite(evidence[name]))
                self.assertEqual(evidence[name], result.metrics[name])
            identity = objective["identity"]
            self.assertEqual(
                identity["cache_content_sha256"],
                hashlib.sha256(fixture.cache_tensor.read_bytes()).hexdigest(),
            )
            self.assertEqual(identity["loss"]["compute_dtype"], "float32")
            self.assertEqual(identity["loss"]["kl_direction"], "teacher||student")

            resumed = root / "resumed"
            self._execute(fixture, resumed, resume=step1)
            with (step2 / "state.pkl").open("rb") as stream:
                expected_state = pickle.load(stream)
            resumed_step2 = resumed / "checkpoints" / "step-00000002"
            with (resumed_step2 / "state.pkl").open("rb") as stream:
                actual_state = pickle.load(stream)
            self.assert_nested_equal(expected_state, actual_state)
            resumed_manifest = json.loads(
                (resumed_step2 / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["metadata"]["objective"],
                resumed_manifest["metadata"]["objective"],
            )
            self.assertEqual(
                manifest["metadata"]["objective_evidence"],
                resumed_manifest["metadata"]["objective_evidence"],
            )

            tampered = root / "tampered-resume"
            shutil.copytree(step1, tampered)
            tampered_manifest_path = tampered / "manifest.json"
            tampered_manifest = json.loads(
                tampered_manifest_path.read_text(encoding="utf-8")
            )
            tampered_manifest["metadata"]["objective"]["identity"][
                "cache_content_sha256"
            ] = "f" * 64
            tampered_manifest_path.write_text(
                json.dumps(tampered_manifest, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "objective/cache identity contract mismatch"
            ):
                self._execute(
                    fixture, root / "tampered-output", resume=tampered
                )

    def test_every_external_identity_changes_fingerprint_and_fails_preflight(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            base = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            base_spec = load_run_spec(fixture.config_path)
            base_resolved, report = resolve_run(
                base_spec, fixture.plugin.manifest, source=fixture.config_path
            )
            self.assertTrue(report.valid, report.issues)
            assert base_resolved is not None
            fields = (
                "cache_content_sha256",
                "teacher_state_sha256",
                "teacher_manifest_sha256",
                "teacher_run_fingerprint",
                "student_state_sha256",
                "student_manifest_sha256",
                "student_run_fingerprint",
                "model_identity_sha256",
                "tokenizer_sha256",
                "processor_sha256",
                "data_sha256",
                "loss_positions_sha256",
            )
            for index, field in enumerate(fields):
                with self.subTest(field=field):
                    value = json.loads(json.dumps(base))
                    value["metadata"]["offline_dense_logit_kd"][field] = (
                        f"{index + 4:x}" * 64
                    )[:64]
                    path = root / f"identity-{field}.json"
                    path.write_text(
                        json.dumps(value, sort_keys=True, indent=2),
                        encoding="utf-8",
                    )
                    spec = load_run_spec(path)
                    resolved, report = resolve_run(
                        spec, fixture.plugin.manifest, source=path
                    )
                    self.assertTrue(report.valid, report.issues)
                    assert resolved is not None
                    self.assertNotEqual(
                        resolved.fingerprint, base_resolved.fingerprint
                    )
                    variant = replace(fixture, config_path=path)
                    with self.assertRaisesRegex(
                        RuntimeError, "expected identity mismatch"
                    ):
                        self._execute(variant, root / f"failed-{field}")

    def test_cache_corruption_and_missing_files_fail_before_step_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            payload = bytearray(fixture.cache_tensor.read_bytes())
            payload[0] ^= 1
            fixture.cache_tensor.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "tensor SHA-256 mismatch"):
                self._execute(fixture, root / "corrupt")
            self.assertFalse(
                (root / "corrupt" / "checkpoints" / "step-00000001").exists()
            )

            fixture = self._fixture(root / "missing-fixture")
            fixture.cache_tensor.unlink()
            with self.assertRaisesRegex(RuntimeError, "identity file is missing"):
                self._execute(fixture, root / "missing-output")

    def test_manifest_positions_must_equal_real_label_mask_before_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            manifest = json.loads(
                fixture.cache_manifest.read_text(encoding="utf-8")
            )
            sample = manifest["samples"][0]
            sample["assistant_positions"] = [2, 4]
            parsed_sample = CacheSampleIdentity.model_validate(sample)
            positions_sha = loss_position_digest((parsed_sample,))
            manifest["loss_positions"]["identity_sha256"] = positions_sha
            fixture.cache_manifest.write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )

            config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            kd = config["metadata"]["offline_dense_logit_kd"]
            kd["cache_manifest_sha256"] = _sha256(fixture.cache_manifest)
            kd["loss_positions_sha256"] = positions_sha
            fixture.config_path.write_text(
                json.dumps(config, sort_keys=True, indent=2), encoding="utf-8"
            )

            forward_calls: list[bool] = []
            original_build = fixture.plugin.build

            def counted_build(model_config):
                bundle = original_build(model_config)
                original_forward = bundle.model.forward

                def counted_forward(*args, **kwargs):
                    forward_calls.append(True)
                    return original_forward(*args, **kwargs)

                bundle.model.forward = counted_forward
                return bundle

            fixture.plugin.build = counted_build
            output = root / "wrong-position-output"
            try:
                with self.assertRaisesRegex(
                    CachedLogitKDError,
                    "assistant target positions differ from labels loss mask",
                ):
                    self._execute(fixture, output)
            finally:
                fixture.plugin.build = original_build
            self.assertEqual(forward_calls, [])
            self.assertFalse(
                (output / "checkpoints" / "step-00000001").exists()
            )

    def test_alignment_shape_dtype_vocab_position_and_digest_fail_closed(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            manifest_data = json.loads(
                fixture.cache_manifest.read_text(encoding="utf-8")
            )
            manifest_data["samples"][0]["input_ids_sha256"] = "f" * 64
            data = manifest_data["data"]
            sample = manifest_data["samples"][0]
            data["identity_sha256"] = canonical_fingerprint(
                {
                    "source_path": data["source_path"],
                    "source_sha256": data["source_sha256"],
                    "split_fingerprint": data["split_fingerprint"],
                    "metadata": data["metadata"],
                    "samples": [
                        {
                            "sample_id": sample["sample_id"],
                            "source_index": sample["source_index"],
                            "canonical_sample_sha256": sample[
                                "canonical_sample_sha256"
                            ],
                            "input_ids_sha256": sample["input_ids_sha256"],
                            "labels_sha256": sample["labels_sha256"],
                        }
                    ],
                }
            )
            fixture.cache_manifest.write_text(
                json.dumps(manifest_data, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            kd = config["metadata"]["offline_dense_logit_kd"]
            kd["cache_manifest_sha256"] = _sha256(fixture.cache_manifest)
            kd["data_sha256"] = data["identity_sha256"]
            fixture.config_path.write_text(
                json.dumps(config, sort_keys=True, indent=2), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CachedLogitKDError, "input_ids digest mismatch"
            ):
                self._execute(fixture, root / "misaligned")

        objective = OfflineDenseLogitKDObjective()
        model = lambda **kwargs: {"logits": torch.zeros((1, 3, 4))}
        plan = BatchPlan(
            items=(BatchItem("s", 0, CostVector(text_tokens=3)),),
            total_cost=CostVector(text_tokens=3),
            budget=BatchBudget(max_samples=1, max_text_tokens=3),
        )

        def batch(
            *,
            teacher_dtype: Any = torch.bfloat16,
            teacher_vocab: int = 4,
            position: int = 1,
            mask_shape: tuple[int, int] = (1, 1),
        ) -> ModelBatch:
            return ModelBatch(
                sample_ids=("s",),
                model_inputs={
                    "labels": torch.tensor([[-100, 1, -100]]),
                    "kd_teacher_logits": torch.zeros(
                        (1, 1, teacher_vocab), dtype=teacher_dtype
                    ),
                    "kd_assistant_positions": torch.tensor([[position]]),
                    "kd_position_mask": torch.ones(mask_shape, dtype=torch.bool),
                    "kd_target_token_ids": torch.tensor([[1]]),
                    "kd_cache_identity": "a" * 64,
                },
                plan=plan,
                trace={
                    "offline_dense_logit_kd": {
                        "cache_content_sha256": "a" * 64,
                        "loss": {
                            "temperature": 2.0,
                            "ce_weight": 0.5,
                            "kd_weight": 0.5,
                        },
                    }
                },
            )

        for value, message in (
            (batch(teacher_dtype=torch.float32), "raw BF16"),
            (batch(teacher_vocab=5), "vocab size mismatch"),
            (batch(position=0), "must be positive"),
            (batch(position=3), "exceeds student sequence"),
            (batch(mask_shape=(1, 2)), "shapes differ"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                CachedLogitKDError, message
            ):
                objective.compute({"model": model}, value)


if __name__ == "__main__":
    unittest.main()
