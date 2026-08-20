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
from trainomni.data import CanonicalSample
from trainomni.models import (
    ComponentCatalog,
    ComponentRule,
    EncodedSample,
    ModelBatch,
    ModelBundle,
    ModelCapabilities,
    ModelPluginManifest,
)
from trainomni.objectives import (
    AssetFileIdentity,
    AssetSetIdentity,
    CachedDPOPairIdentity,
    CheckpointIdentity,
    DPOAlgorithmIdentity,
    DPOBatchPairIdentity,
    DPOBranchIdentity,
    DPOCacheIdentity,
    DPODataIdentity,
    ModelIdentity,
    OfflineDPOPreferenceManifest,
    OfflineReferenceDPOCacheManifest,
    OfflineReferenceDPOError,
    OfflineReferenceDPOObjective,
    PreferenceManifestIdentity,
    PreferencePairIdentity,
    ReferenceLogProbTensorIdentity,
    asset_set_digest,
    dpo_data_identity_digest,
    integer_tensor_digest,
    model_inputs_digest,
    preference_manifest_digest,
    preference_pair_digest,
)
from trainomni.runtime import StageRunRequest, execute_stage

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
BASE_CONFIG = FRAMEWORK_ROOT / "configs" / "examples" / "torch_toy_smoke.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(value: Any) -> bytes:
    import torch

    return value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()


class _DPOToyPlugin:
    manifest = ModelPluginManifest(
        plugin_id="torch-toy-dpo",
        plugin_version="1.0.0",
        capabilities=ModelCapabilities(
            modalities=frozenset({"text"}),
            content_blocks=frozenset({"text"}),
            objectives=frozenset({"preference"}),
            max_media_per_sample=0,
            supports_packing=False,
            supports_generation=False,
            attention_backends=frozenset({"eager"}),
            parallelism=frozenset({"single"}),
            engine_backends=frozenset({"torch"}),
            export_formats=frozenset({"torch"}),
        ),
        component_ids=("vision_encoder", "connector", "language_model"),
        model_patterns=("torch-toy-dpo/*",),
        dependency_constraints=("torch>=2.4",),
    )

    def __init__(self) -> None:
        self.pair_identities: dict[str, dict[str, Any]] = {}
        self.forward_calls = 0
        self.common_mismatch = False

    def capabilities(self):
        return self.manifest.capabilities

    def build(self, config: Mapping[str, Any]) -> ModelBundle:
        import torch

        plugin = self

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision = torch.nn.Linear(1, 8)
                self.connector = torch.nn.Linear(8, 8)
                self.language = torch.nn.Embedding(64, 8)
                self.lm_head = torch.nn.Linear(8, 64)

            def forward(self, input_ids, labels, **kwargs):
                plugin.forward_calls += 1
                hidden = self.language(input_ids)
                vision = self.vision(
                    torch.ones((*hidden.shape[:-1], 1), device=hidden.device)
                )
                connected = self.connector(hidden + vision)
                logits = self.lm_head(connected)
                return {"logits": logits}

        torch.manual_seed(1234)
        return ModelBundle(ToyModel(), metadata={"kind": "torch-toy-dpo"})

    def component_catalog(self, bundle):
        return ComponentCatalog(
            (
                ComponentRule("vision_encoder", ("vision.",)),
                ComponentRule("connector", ("connector.",)),
                ComponentRule("language_model", ("language.", "lm_head.")),
            )
        )

    def validate_sample(self, sample: CanonicalSample, objective: str):
        return ()

    def encode(self, sample: CanonicalSample, context: Any) -> EncodedSample:
        chosen_ids = [5, 2, 3, 4]
        rejected_ids = [5, 2, 3, 5]
        common = {
            "attention_mask": [1, 1, 1, 1],
            "media_features": [0.25, 0.5],
        }
        rejected_common = dict(common)
        if self.common_mismatch:
            rejected_common["media_features"] = [0.25, 0.75]
        return EncodedSample(
            sample_id=sample.id,
            model_inputs={
                "chosen": {
                    "input_ids": chosen_ids,
                    "labels": [-100, -100, 3, 4],
                    **common,
                },
                "rejected": {
                    "input_ids": rejected_ids,
                    "labels": [-100, -100, 3, 5],
                    **rejected_common,
                },
                "dpo_pair_identity": self.pair_identities[sample.id],
            },
            cost=CostVector(text_tokens=8),
        )

    def collate(self, samples, plan: BatchPlan) -> ModelBatch:
        import torch

        if len(samples) != 1:
            raise ValueError("toy DPO plugin requires batch size 1")
        sample = samples[0]

        def branch(name: str) -> dict[str, Any]:
            value = sample.model_inputs[name]
            return {
                "input_ids": torch.tensor([value["input_ids"]], dtype=torch.long),
                "labels": torch.tensor([value["labels"]], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [value["attention_mask"]], dtype=torch.long
                ),
                "media_features": torch.tensor(
                    [value["media_features"]], dtype=torch.float32
                ),
            }

        return ModelBatch(
            sample_ids=(sample.sample_id,),
            model_inputs={
                "chosen": branch("chosen"),
                "rejected": branch("rejected"),
                "dpo_pair_identity": sample.model_inputs["dpo_pair_identity"],
            },
            plan=plan,
        )

    def export(self, bundle, checkpoint, target):
        raise NotImplementedError


@dataclass(slots=True)
class _DPOFixture:
    config_path: Path
    plugin: _DPOToyPlugin
    checkpoint: Path
    preference_manifest: Path
    cache_manifest: Path
    chosen_tensor: Path
    rejected_tensor: Path
    artifact: ArtifactRef


@unittest.skipUnless(TORCH_AVAILABLE, "requires optional torch runtime")
class OfflineReferenceDPOTests(unittest.TestCase):
    def _checkpoint(self, root: Path, plugin: _DPOToyPlugin, run: str) -> Path:
        bundle = plugin.build({})
        registry = StateRegistry()
        registry.register("model", ObjectState(bundle.model))
        return LocalCheckpointManager(root).save(
            "step-00000000", registry, metadata={"run_fingerprint": run}
        )

    def _fixture(self, root: Path) -> _DPOFixture:
        import torch

        plugin = _DPOToyPlugin()
        producer_run = "1" * 64
        checkpoint = self._checkpoint(root / "policy", plugin, producer_run)
        artifact = ArtifactRef(
            "policy", selector="step-00000000", uri=str(checkpoint)
        )
        plugin.forward_calls = 0

        dataset_path = root / "preference.jsonl"
        canonical = {
            "schema_version": "trainomni.sample.v0.1",
            "id": "dpo-toy-001",
            "objective": "preference",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Choose A or B"}],
                }
            ],
            "preference": {
                "chosen": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Answer: A"}],
                        }
                    ],
                    "score": 1.0,
                },
                "rejected": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Answer: B"}],
                        }
                    ],
                    "score": 0.0,
                },
                "margin": 1.0,
                "judge": "synthetic-cyclic-v1",
            },
        }
        dataset_path.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
        source_sha = _sha256(dataset_path)
        data_base = DPODataIdentity.model_construct(
            source_path=str(dataset_path),
            source_sha256=source_sha,
            split_fingerprints={"train": "2" * 64},
            metadata={"reader": "canonical", "seed": 20260820},
            identity_sha256="0" * 64,
        )
        data = DPODataIdentity(
            **data_base.model_dump(exclude={"identity_sha256"}),
            identity_sha256=dpo_data_identity_digest(data_base),
        )
        pair_base = PreferencePairIdentity.model_construct(
            sample_id="dpo-toy-001",
            source_index=0,
            split="train",
            order_index=0,
            canonical_pair_sha256="3" * 64,
            common_prompt_sha256="4" * 64,
            media_sha256="5" * 64,
            chosen_canonical_sha256="6" * 64,
            rejected_canonical_sha256="7" * 64,
            construction_rule="cyclic-A-to-B-v1",
            judge="synthetic-cyclic-v1",
            chosen_score=1.0,
            rejected_score=0.0,
            margin=1.0,
            identity_sha256="0" * 64,
        )
        pair = PreferencePairIdentity(
            **pair_base.model_dump(exclude={"identity_sha256"}),
            identity_sha256=preference_pair_digest(pair_base),
        )
        preference_base = OfflineDPOPreferenceManifest.model_construct(
            preference_id="toy-preference-v1",
            producer_code_revision="test-revision",
            data=data,
            pairs=(pair,),
            identity_sha256="0" * 64,
        )
        preference = OfflineDPOPreferenceManifest(
            **preference_base.model_dump(exclude={"identity_sha256"}),
            identity_sha256=preference_manifest_digest(preference_base),
        )
        cache_root = root / "cache"
        cache_root.mkdir()
        preference_path = cache_root / "preference-manifest.json"
        preference_path.write_text(preference.model_dump_json(indent=2), encoding="utf-8")

        chosen_ids = torch.tensor([5, 2, 3, 4], dtype=torch.long)
        chosen_labels = torch.tensor([-100, -100, 3, 4], dtype=torch.long)
        rejected_ids = torch.tensor([5, 2, 3, 5], dtype=torch.long)
        rejected_labels = torch.tensor([-100, -100, 3, 5], dtype=torch.long)
        common_inputs = {
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "media_features": torch.tensor([[0.25, 0.5]], dtype=torch.float32),
        }
        common_inputs_sha = model_inputs_digest(common_inputs)
        chosen_logps = torch.tensor([-1.25, -1.5], dtype=torch.float32)
        rejected_logps = torch.tensor([-1.75, -1.125], dtype=torch.float32)
        chosen_payload = _tensor_bytes(chosen_logps)
        rejected_payload = _tensor_bytes(rejected_logps)
        chosen_path = cache_root / "dpo-toy-001.chosen.fp32"
        rejected_path = cache_root / "dpo-toy-001.rejected.fp32"
        chosen_path.write_bytes(chosen_payload)
        rejected_path.write_bytes(rejected_payload)

        def branch(
            canonical_sha: str,
            ids: Any,
            labels: Any,
            targets: tuple[int, ...],
            path: Path,
            payload: bytes,
            logps: Any,
        ) -> DPOBranchIdentity:
            return DPOBranchIdentity(
                canonical_sha256=canonical_sha,
                input_ids_sha256=integer_tensor_digest(ids),
                labels_sha256=integer_tensor_digest(labels),
                assistant_positions=(2, 3),
                causal_positions=(1, 2),
                target_token_ids=targets,
                reference_logps=ReferenceLogProbTensorIdentity(
                    file=path.name,
                    shape=(2,),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    sequence_logp=float(logps.sum(dtype=torch.float32).item()),
                ),
            )

        cached_pair = CachedDPOPairIdentity(
            sample_id=pair.sample_id,
            source_index=pair.source_index,
            split=pair.split,
            order_index=pair.order_index,
            preference_pair_sha256=pair.identity_sha256,
            canonical_pair_sha256=pair.canonical_pair_sha256,
            common_prompt_sha256=pair.common_prompt_sha256,
            media_sha256=pair.media_sha256,
            common_model_inputs_sha256=common_inputs_sha,
            chosen=branch(
                pair.chosen_canonical_sha256,
                chosen_ids,
                chosen_labels,
                (3, 4),
                chosen_path,
                chosen_payload,
                chosen_logps,
            ),
            rejected=branch(
                pair.rejected_canonical_sha256,
                rejected_ids,
                rejected_labels,
                (3, 5),
                rejected_path,
                rejected_payload,
                rejected_logps,
            ),
        )
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

        policy = CheckpointIdentity(
            artifact=str(artifact),
            path=str(checkpoint),
            state_sha256=_sha256(checkpoint / "state.pkl"),
            manifest_sha256=_sha256(checkpoint / "manifest.json"),
            run_fingerprint=producer_run,
        )
        model_assets = assets("model")
        tokenizer = assets("tokenizer")
        processor = assets("processor")
        content_sha = hashlib.sha256(chosen_payload + rejected_payload).hexdigest()
        cache = OfflineReferenceDPOCacheManifest(
            cache_id="toy-offline-reference-dpo-v1",
            producer_code_revision="test-revision",
            reference=policy,
            policy=policy,
            model=ModelIdentity(
                plugin_id=plugin.manifest.plugin_id,
                plugin_version=plugin.manifest.plugin_version,
                assets=model_assets,
            ),
            tokenizer=tokenizer,
            processor=processor,
            preference_manifest=PreferenceManifestIdentity(
                path=str(preference_path),
                sha256=_sha256(preference_path),
                identity_sha256=preference.identity_sha256,
            ),
            data_identity_sha256=data.identity_sha256,
            pair_identity_sha256=canonical_fingerprint([pair.identity_sha256]),
            algorithm=DPOAlgorithmIdentity(),
            logps=DPOCacheIdentity(
                vocab_size=64,
                pair_count=1,
                total_positions=4,
                total_bytes=len(chosen_payload) + len(rejected_payload),
                content_sha256=content_sha,
            ),
            pairs=(cached_pair,),
        )
        cache_path = cache_root / "manifest.json"
        cache_path.write_text(cache.model_dump_json(indent=2), encoding="utf-8")
        plugin.pair_identities[pair.sample_id] = DPOBatchPairIdentity(
            sample_id=pair.sample_id,
            preference_pair_sha256=pair.identity_sha256,
            canonical_pair_sha256=pair.canonical_pair_sha256,
            common_prompt_sha256=pair.common_prompt_sha256,
            media_sha256=pair.media_sha256,
            common_model_inputs_sha256=common_inputs_sha,
            chosen_canonical_sha256=pair.chosen_canonical_sha256,
            rejected_canonical_sha256=pair.rejected_canonical_sha256,
        ).model_dump(mode="json")

        config = load_run_spec(BASE_CONFIG).model_dump(mode="json")
        config["name"] = "torch_toy_offline_reference_dpo"
        config["model"] = {"plugin": plugin.manifest.plugin_id, "config": {}}
        stage = config["stage"]
        stage.update(
            {
                "stage_id": "offline_reference_dpo",
                "stage_type": "offline_preference",
                "objective": "preference",
                "objective_impl": "offline-reference-dpo",
                "inputs": {"model": str(artifact)},
            }
        )
        stage["data"]["datasets"][0]["uri"] = str(dataset_path)
        stage["component_policy"] = {
            "vision_encoder": {"trainable": False, "dtype": "fp32"},
            "connector": {
                "trainable": True,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dtype": "fp32",
                "gradient_clip": 1.0,
            },
            "language_model": {"trainable": False, "dtype": "fp32"},
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
        stage["engine"].update({"precision": "fp32", "config": {"device": "cpu"}})
        stage["checkpoint"].update({"every_steps": 1})
        config["metadata"] = {
            "offline_reference_dpo": {
                "cache_manifest": str(cache_path),
                "cache_manifest_sha256": _sha256(cache_path),
                "cache_content_sha256": content_sha,
                "preference_manifest_sha256": _sha256(preference_path),
                "preference_identity_sha256": preference.identity_sha256,
                "pair_identity_sha256": cache.pair_identity_sha256,
                "data_identity_sha256": data.identity_sha256,
                "reference_state_sha256": policy.state_sha256,
                "reference_manifest_sha256": policy.manifest_sha256,
                "reference_run_fingerprint": policy.run_fingerprint,
                "policy_state_sha256": policy.state_sha256,
                "policy_manifest_sha256": policy.manifest_sha256,
                "policy_run_fingerprint": policy.run_fingerprint,
                "model_identity_sha256": model_assets.identity_sha256,
                "tokenizer_sha256": tokenizer.identity_sha256,
                "processor_sha256": processor.identity_sha256,
                "beta": 0.1,
                "loss_variant": "sigmoid",
                "label_smoothing": 0.0,
                "sequence_reduction": "sum",
                "pair_reduction": "mean",
                "compute_dtype": "float32",
                "reference_free": False,
                "auxiliary_ce": False,
                "expected_pair_count": 1,
                "expected_total_positions": 4,
                "vocab_size": 64,
                "max_cache_bytes": 1024,
            }
        }
        config_path = root / "dpo.json"
        config_path.write_text(
            json.dumps(config, sort_keys=True, indent=2), encoding="utf-8"
        )
        return _DPOFixture(
            config_path=config_path,
            plugin=plugin,
            checkpoint=checkpoint,
            preference_manifest=preference_path,
            cache_manifest=cache_path,
            chosen_tensor=chosen_path,
            rejected_tensor=rejected_path,
            artifact=artifact,
        )

    def _execute(
        self,
        fixture: _DPOFixture,
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

    def test_fp32_sigmoid_dpo_oracle_live_reference_and_two_branch_gradient(self) -> None:
        import torch

        class Student(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.connector = torch.nn.Linear(2, 2, bias=False)
                self.language = torch.nn.Linear(2, 4, bias=False)
                for parameter in self.language.parameters():
                    parameter.requires_grad_(False)
                self.branch_outputs: list[Any] = []

            def forward(self, features, labels):
                connected = self.connector(features)
                connected.retain_grad()
                self.branch_outputs.append(connected)
                return {"logits": self.language(connected)}

        torch.manual_seed(7)
        model = Student()
        chosen_features = torch.tensor([[[0.2, 0.4], [0.3, -0.2], [0.5, 0.1]]])
        rejected_features = torch.tensor([[[0.2, 0.4], [-0.1, 0.6], [0.4, -0.3]]])
        chosen_targets = torch.tensor([[1, 2]])
        rejected_targets = torch.tensor([[1, 3]])
        positions = torch.tensor([[1, 2]])
        ref_chosen = torch.tensor([[-1.2, -1.4]], dtype=torch.float32)
        ref_rejected = torch.tensor([[-1.5, -1.1]], dtype=torch.float32)
        plan = BatchPlan(
            items=(BatchItem("pair", 0, CostVector(text_tokens=6)),),
            total_cost=CostVector(text_tokens=6),
            budget=BatchBudget(max_samples=1, max_text_tokens=8),
        )
        batch = ModelBatch(
            sample_ids=("pair",),
            model_inputs={
                "dpo_chosen_inputs": {
                    "features": chosen_features,
                    "labels": torch.tensor([[-100, 1, 2]]),
                },
                "dpo_rejected_inputs": {
                    "features": rejected_features,
                    "labels": torch.tensor([[-100, 1, 3]]),
                },
                "dpo_chosen_positions": positions,
                "dpo_rejected_positions": positions,
                "dpo_chosen_targets": chosen_targets,
                "dpo_rejected_targets": rejected_targets,
                "dpo_reference_chosen_logps": ref_chosen,
                "dpo_reference_rejected_logps": ref_rejected,
                "dpo_cache_identity": "a" * 64,
                "dpo_pair_identity": "b" * 64,
            },
            plan=plan,
            trace={
                "offline_reference_dpo": {
                    "cache_content_sha256": "a" * 64,
                    "preference_pair_sha256": "b" * 64,
                    "algorithm": DPOAlgorithmIdentity().model_dump(mode="json"),
                }
            },
        )
        objective = OfflineReferenceDPOObjective()
        self.assertEqual(
            objective.manifest.requirements.sample_objectives,
            frozenset({"preference"}),
        )
        self.assertFalse(objective.manifest.requirements.requires_reference_model)
        self.assertEqual(objective.manifest.supported_engines, frozenset({"torch"}))
        with self.assertRaisesRegex(OfflineReferenceDPOError, "live reference"):
            objective.compute({"model": model, "reference": object()}, batch)
        output = objective.compute({"model": model}, batch)

        chosen_logits = model.language(model.connector(chosen_features)).float()
        rejected_logits = model.language(model.connector(rejected_features)).float()
        chosen_policy = torch.nn.functional.log_softmax(
            chosen_logits[0, positions[0] - 1], dim=-1
        ).gather(1, chosen_targets[0].unsqueeze(1)).sum()
        rejected_policy = torch.nn.functional.log_softmax(
            rejected_logits[0, positions[0] - 1], dim=-1
        ).gather(1, rejected_targets[0].unsqueeze(1)).sum()
        rho_policy = chosen_policy - rejected_policy
        rho_reference = ref_chosen.sum() - ref_rejected.sum()
        expected_delta = rho_policy - rho_reference
        expected_logit = 0.1 * expected_delta
        expected_loss = torch.nn.functional.softplus(-expected_logit)
        self.assertEqual(output.total.dtype, torch.float32)
        self.assertTrue(torch.allclose(output.total, expected_loss))
        self.assertAlmostEqual(output.metrics["delta"], expected_delta.item())
        self.assertAlmostEqual(output.metrics["dpo_logit"], expected_logit.item())
        self.assertAlmostEqual(
            output.metrics["reward_margin"], output.metrics["dpo_logit"]
        )
        self.assertEqual(output.counts["preference_pairs"], 1)
        self.assertEqual(output.counts["loss_tokens"], 4)
        output.total.backward()
        self.assertGreater(model.connector.weight.grad.norm().item(), 0)
        self.assertIsNone(model.language.weight.grad)
        self.assertEqual(len(model.branch_outputs), 2)
        self.assertTrue(all(value.grad is not None for value in model.branch_outputs))
        self.assertTrue(all(value.grad.norm().item() > 0 for value in model.branch_outputs))

    def test_cache_e2e_evidence_counts_checkpoint_and_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            first = root / "first"
            result = self._execute(fixture, first)
            self.assertEqual(result.status, "succeeded")
            expected_metrics = (
                "policy_chosen_logp",
                "policy_rejected_logp",
                "reference_chosen_logp",
                "reference_rejected_logp",
                "policy_log_ratio",
                "reference_log_ratio",
                "delta",
                "dpo_logit",
                "reward_chosen",
                "reward_rejected",
                "reward_margin",
                "preference_accuracy",
                "loss",
                "chosen_target_tokens",
                "rejected_target_tokens",
                "pair_count",
            )
            self.assertTrue(all(math.isfinite(result.metrics[key]) for key in expected_metrics))
            step1 = first / "checkpoints" / "step-00000001"
            step2 = first / "checkpoints" / "step-00000002"
            manifest = json.loads((step2 / "manifest.json").read_text(encoding="utf-8"))
            metadata = manifest["metadata"]
            self.assertEqual(metadata["objective"]["objective_id"], "offline-reference-dpo")
            self.assertEqual(
                metadata["objective"]["state_count_keys"],
                [
                    "preference_pairs",
                    "chosen_loss_tokens",
                    "rejected_loss_tokens",
                ],
            )
            self.assertEqual(
                metadata["objective_counts"],
                {
                    "preference_pairs": 2,
                    "chosen_loss_tokens": 4,
                    "rejected_loss_tokens": 4,
                },
            )
            self.assertTrue(all(key in metadata["objective_evidence"] for key in expected_metrics))
            with (step2 / "state.pkl").open("rb") as stream:
                expected_state = pickle.load(stream)
            self.assertEqual(
                expected_state["objects"]["objective_count_preference_pairs"]["value"],
                2,
            )

            resumed = root / "resumed"
            self._execute(fixture, resumed, resume=step1)
            resumed_step2 = resumed / "checkpoints" / "step-00000002"
            with (resumed_step2 / "state.pkl").open("rb") as stream:
                actual_state = pickle.load(stream)
            self.assert_nested_equal(expected_state, actual_state)
            resumed_manifest = json.loads(
                (resumed_step2 / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["objective"], resumed_manifest["metadata"]["objective"])
            self.assertEqual(
                metadata["objective_counts"],
                resumed_manifest["metadata"]["objective_counts"],
            )

            tampered = root / "tampered"
            shutil.copytree(step1, tampered)
            tampered_manifest_path = tampered / "manifest.json"
            value = json.loads(tampered_manifest_path.read_text(encoding="utf-8"))
            value["metadata"]["objective"]["identity"]["cache_content_sha256"] = "f" * 64
            tampered_manifest_path.write_text(
                json.dumps(value, sort_keys=True, indent=2), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "objective/cache identity contract mismatch"
            ):
                self._execute(fixture, root / "tampered-output", resume=tampered)

    def test_external_identities_change_fingerprint_and_fail_preflight(self) -> None:
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
                "preference_manifest_sha256",
                "preference_identity_sha256",
                "pair_identity_sha256",
                "data_identity_sha256",
                "reference_state_sha256",
                "reference_manifest_sha256",
                "reference_run_fingerprint",
                "policy_state_sha256",
                "policy_manifest_sha256",
                "policy_run_fingerprint",
                "model_identity_sha256",
                "tokenizer_sha256",
                "processor_sha256",
            )
            for index, field in enumerate(fields):
                with self.subTest(field=field):
                    value = json.loads(json.dumps(base))
                    value["metadata"]["offline_reference_dpo"][field] = (
                        f"{index + 1:x}" * 64
                    )[:64]
                    path = root / f"identity-{field}.json"
                    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
                    spec = load_run_spec(path)
                    resolved, report = resolve_run(
                        spec, fixture.plugin.manifest, source=path
                    )
                    self.assertTrue(report.valid, report.issues)
                    assert resolved is not None
                    self.assertNotEqual(resolved.fingerprint, base_resolved.fingerprint)
                    with self.assertRaisesRegex(RuntimeError, "expected identity mismatch"):
                        self._execute(replace(fixture, config_path=path), root / f"bad-{field}")

            value = json.loads(json.dumps(base))
            value["metadata"]["offline_reference_dpo"]["beta"] = 0.2
            path = root / "bad-beta.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid offline DPO config"):
                self._execute(replace(fixture, config_path=path), root / "bad-beta")

            value = json.loads(json.dumps(base))
            value["stage"]["stage_type"] = "instruction_sft"
            path = root / "bad-stage.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "requires stage_type"):
                self._execute(replace(fixture, config_path=path), root / "bad-stage")

    def test_corruption_and_self_consistent_pair_swap_fail_before_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root / "corrupt")
            payload = bytearray(fixture.chosen_tensor.read_bytes())
            payload[0] ^= 1
            fixture.chosen_tensor.write_bytes(payload)
            fixture.plugin.forward_calls = 0
            output = root / "corrupt-output"
            with self.assertRaisesRegex(RuntimeError, "tensor SHA-256 mismatch"):
                self._execute(fixture, output)
            self.assertEqual(fixture.plugin.forward_calls, 0)
            self.assertFalse((output / "checkpoints" / "step-00000001").exists())

            fixture = self._fixture(root / "missing")
            fixture.rejected_tensor.unlink()
            fixture.plugin.forward_calls = 0
            output = root / "missing-output"
            with self.assertRaisesRegex(RuntimeError, "identity file is missing"):
                self._execute(fixture, output)
            self.assertEqual(fixture.plugin.forward_calls, 0)
            self.assertFalse((output / "checkpoints" / "step-00000001").exists())

            for variant, mutation in (
                ("dtype", {"dtype": "bfloat16"}),
                ("shape", {"shape": [1]}),
            ):
                fixture = self._fixture(root / variant)
                manifest = json.loads(
                    fixture.cache_manifest.read_text(encoding="utf-8")
                )
                manifest["pairs"][0]["chosen"]["reference_logps"].update(
                    mutation
                )
                fixture.cache_manifest.write_text(
                    json.dumps(manifest, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
                config = json.loads(
                    fixture.config_path.read_text(encoding="utf-8")
                )
                config["metadata"]["offline_reference_dpo"][
                    "cache_manifest_sha256"
                ] = _sha256(fixture.cache_manifest)
                fixture.config_path.write_text(
                    json.dumps(config, sort_keys=True), encoding="utf-8"
                )
                fixture.plugin.forward_calls = 0
                output = root / f"{variant}-output"
                with self.assertRaisesRegex(RuntimeError, "invalid DPO cache manifest"):
                    self._execute(fixture, output)
                self.assertEqual(fixture.plugin.forward_calls, 0)
                self.assertFalse(
                    (output / "checkpoints" / "step-00000001").exists()
                )

            fixture = self._fixture(root / "target")
            manifest = json.loads(fixture.cache_manifest.read_text(encoding="utf-8"))
            manifest["pairs"][0]["chosen"]["target_token_ids"] = [3, 6]
            fixture.cache_manifest.write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )
            config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            config["metadata"]["offline_reference_dpo"][
                "cache_manifest_sha256"
            ] = _sha256(fixture.cache_manifest)
            fixture.config_path.write_text(json.dumps(config), encoding="utf-8")
            fixture.plugin.forward_calls = 0
            output = root / "target-output"
            with self.assertRaisesRegex(
                OfflineReferenceDPOError, "target-token alignment mismatch"
            ):
                self._execute(fixture, output)
            self.assertEqual(fixture.plugin.forward_calls, 0)
            self.assertFalse((output / "checkpoints" / "step-00000001").exists())

            fixture = self._fixture(root / "swap")
            manifest = json.loads(fixture.cache_manifest.read_text(encoding="utf-8"))
            pair = manifest["pairs"][0]
            pair["chosen"], pair["rejected"] = pair["rejected"], pair["chosen"]
            content = fixture.rejected_tensor.read_bytes() + fixture.chosen_tensor.read_bytes()
            manifest["logps"]["content_sha256"] = hashlib.sha256(content).hexdigest()
            fixture.cache_manifest.write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )
            config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            kd = config["metadata"]["offline_reference_dpo"]
            kd["cache_manifest_sha256"] = _sha256(fixture.cache_manifest)
            kd["cache_content_sha256"] = manifest["logps"]["content_sha256"]
            fixture.config_path.write_text(
                json.dumps(config, sort_keys=True, indent=2), encoding="utf-8"
            )
            fixture.plugin.forward_calls = 0
            output = root / "swap-output"
            with self.assertRaisesRegex(RuntimeError, "cache/preference pair mismatch"):
                self._execute(fixture, output)
            self.assertEqual(fixture.plugin.forward_calls, 0)
            self.assertFalse((output / "checkpoints" / "step-00000001").exists())

    def test_label_positions_common_media_and_identical_pair_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "chosen and rejected canonical"):
            base = PreferencePairIdentity.model_construct(
                sample_id="x",
                source_index=0,
                split="train",
                order_index=0,
                canonical_pair_sha256="1" * 64,
                common_prompt_sha256="2" * 64,
                media_sha256="3" * 64,
                chosen_canonical_sha256="4" * 64,
                rejected_canonical_sha256="4" * 64,
                construction_rule="rule",
                judge="judge",
                identity_sha256="0" * 64,
            )
            PreferencePairIdentity(
                **base.model_dump(exclude={"identity_sha256"}),
                identity_sha256=preference_pair_digest(base),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root / "positions")
            manifest = json.loads(fixture.cache_manifest.read_text(encoding="utf-8"))
            for name in ("chosen", "rejected"):
                branch = manifest["pairs"][0][name]
                branch["assistant_positions"] = [1, 3]
                branch["causal_positions"] = [0, 2]
            fixture.cache_manifest.write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )
            config = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            config["metadata"]["offline_reference_dpo"]["cache_manifest_sha256"] = (
                _sha256(fixture.cache_manifest)
            )
            fixture.config_path.write_text(json.dumps(config), encoding="utf-8")
            fixture.plugin.forward_calls = 0
            output = root / "positions-output"
            with self.assertRaisesRegex(
                OfflineReferenceDPOError, "positions differ from real labels loss mask"
            ):
                self._execute(fixture, output)
            self.assertEqual(fixture.plugin.forward_calls, 0)
            self.assertFalse((output / "checkpoints" / "step-00000001").exists())

            fixture = self._fixture(root / "common")
            fixture.plugin.common_mismatch = True
            fixture.plugin.forward_calls = 0
            output = root / "common-output"
            with self.assertRaisesRegex(
                OfflineReferenceDPOError, "common media/model inputs differ"
            ):
                self._execute(fixture, output)
            self.assertEqual(fixture.plugin.forward_calls, 0)
            self.assertFalse((output / "checkpoints" / "step-00000001").exists())


if __name__ == "__main__":
    unittest.main()
