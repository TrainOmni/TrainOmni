from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.config import (
    DatasetSpec,
    DataSpec,
    EngineSpec,
    OptimizationSpec,
    StageSpec,
)
from trainomni.contracts import BatchBudget
from trainomni.data import (
    DistributedBatchStream,
    MixtureStream,
    StatefulBatchStream,
    open_dataset_streams,
)
from trainomni.data.batching import BatchPlanningError
from trainomni.evaluation import (
    EvaluationRequest,
    LossEvaluator,
)
from trainomni.models import ModelBundle
from trainomni.objectives import ObjectiveRegistry, resolve_objective


def load_toy_plugin():
    path = FRAMEWORK_ROOT / "tests" / "plugins" / "toy_vlm_plugin.py"
    spec = importlib.util.spec_from_file_location("runtime_toy_plugin", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PLUGIN


def sample(sample_id: str, words: str) -> dict:
    return {
        "schema_version": "trainomni.sample.v0.1",
        "id": sample_id,
        "objective": "sft",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
            {"role": "assistant", "content": [{"type": "text", "text": words}]},
        ],
    }


class RuntimeDataTests(unittest.TestCase):
    def _streams(self, root: Path):
        paths = []
        for dataset, records in (
            ("a", [sample("a1", "one"), sample("a2", "two two")]),
            ("b", [sample("b1", "three"), sample("b2", "four four")]),
        ):
            path = root / f"{dataset}.jsonl"
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in records),
                encoding="utf-8",
            )
            paths.append(
                DatasetSpec(
                    dataset_id=dataset,
                    uri=str(path),
                    importer="canonical",
                    weight=1.0 if dataset == "a" else 2.0,
                )
            )
        return open_dataset_streams(tuple(paths), source_config=None)

    def test_weighted_mixture_exact_resume_and_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mixture = MixtureStream(self._streams(root), seed=11, repeat=True)
            prefix = [next(mixture).sample.id for _ in range(3)]
            state = mixture.state_dict()
            expected = [next(mixture).sample.id for _ in range(8)]

            restored = MixtureStream(self._streams(root), seed=11, repeat=True)
            restored.load_state_dict(state)
            actual = [next(restored).sample.id for _ in range(8)]
            self.assertEqual(actual, expected)
            self.assertEqual(len(prefix), 3)

    def test_batch_stream_preserves_lookahead_on_resume(self) -> None:
        plugin = load_toy_plugin()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mixture = MixtureStream(self._streams(root), seed=7, repeat=True)
            batches = StatefulBatchStream(
                mixture,
                plugin=plugin,
                sample_objective="sft",
                stage_id="sft",
                budget=BatchBudget(max_samples=2, max_text_tokens=7),
                packing=False,
            )
            first = next(batches)
            state = batches.state_dict()
            expected = [next(batches).sample_ids for _ in range(5)]

            restored = StatefulBatchStream(
                MixtureStream(self._streams(root), seed=7, repeat=True),
                plugin=plugin,
                sample_objective="sft",
                stage_id="sft",
                budget=BatchBudget(max_samples=2, max_text_tokens=7),
                packing=False,
            )
            restored.load_state_dict(state)
            actual = [next(restored).sample_ids for _ in range(5)]
            self.assertEqual(actual, expected)
            self.assertTrue(first.sample_ids)

    def test_data_spec_resume_identity_canonicalizes_only_set_fields(self) -> None:
        plugin = load_toy_plugin()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = tuple(item.spec for item in self._streams(root))
            data_spec = DataSpec(
                datasets=datasets,
                modalities=frozenset({"text", "image"}),
                content_blocks=frozenset({"text", "media"}),
                max_media_per_sample=1,
                config={"max_samples": 1, "repeat": True},
            )

            def stream() -> StatefulBatchStream:
                return StatefulBatchStream(
                    MixtureStream(self._streams(root), seed=7, repeat=True),
                    plugin=plugin,
                    sample_objective="sft",
                    stage_id="sft",
                    budget=BatchBudget(max_samples=1),
                    packing=False,
                    data_spec=data_spec,
                )

            original = stream()
            next(original)
            state = original.state_dict()
            self.assertEqual(state["data_spec"]["modalities"], ["image", "text"])
            self.assertEqual(
                state["data_spec"]["content_blocks"], ["media", "text"]
            )

            reversed_sets = copy.deepcopy(state)
            reversed_sets["data_spec"]["modalities"].reverse()
            reversed_sets["data_spec"]["content_blocks"].reverse()
            restored = stream()
            restored.load_state_dict(reversed_sets)
            self.assertEqual(next(restored).sample_ids, next(original).sample_ids)

            reversed_datasets = copy.deepcopy(state)
            reversed_datasets["data_spec"]["datasets"].reverse()
            with self.assertRaisesRegex(BatchPlanningError, "data_spec mismatch"):
                stream().load_state_dict(reversed_datasets)

            code = (
                "import json,sys;"
                f"sys.path.insert(0,{str(SRC_ROOT)!r});"
                "from trainomni.config import DataSpec;"
                "from trainomni.data.runtime import _data_spec_state;"
                "value=DataSpec(modalities=frozenset({'text','image'}),"
                "content_blocks=frozenset({'text','media'}),max_media_per_sample=1);"
                "print(json.dumps(_data_spec_state(value),sort_keys=True))"
            )
            outputs = []
            for seed in ("1", "31337"):
                environment = os.environ.copy()
                environment["PYTHONHASHSEED"] = seed
                outputs.append(
                    subprocess.check_output(
                        [sys.executable, "-c", code],
                        cwd=FRAMEWORK_ROOT,
                        env=environment,
                        text=True,
                    ).strip()
                )
            self.assertEqual(outputs[0], outputs[1])

    def test_distributed_batch_grouping_is_rank_deterministic(self) -> None:
        plugin = load_toy_plugin()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def rank_stream(rank: int):
                base = StatefulBatchStream(
                    MixtureStream(self._streams(root), seed=5, repeat=True),
                    plugin=plugin,
                    sample_objective="sft",
                    stage_id="sft",
                    budget=BatchBudget(max_samples=1),
                    packing=False,
                )
                return DistributedBatchStream(base, rank=rank, world_size=2)

            rank0 = rank_stream(0)
            rank1 = rank_stream(1)
            values0 = [next(rank0).sample_ids for _ in range(4)]
            values1 = [next(rank1).sample_ids for _ in range(4)]
            self.assertNotEqual(values0, values1)
            self.assertEqual(
                rank0.state_dict()["batches"]["mixture"]["draw_count"],
                rank1.state_dict()["batches"]["mixture"]["draw_count"],
            )

    def test_objective_binding_separates_sample_semantics(self) -> None:
        stage = StageSpec(
            stage_id="sft",
            stage_type="instruction_sft",
            objective="sft",
            data=DataSpec(
                datasets=(DatasetSpec(dataset_id="x", uri="x.jsonl", importer="canonical"),)
            ),
            optimization=OptimizationSpec(max_steps=1),
            engine=EngineSpec(backend="torch", precision="fp32"),
        )
        binding = resolve_objective(stage, ObjectiveRegistry())
        self.assertEqual(binding.sample_objective, "sft")
        self.assertEqual(binding.implementation_id, "masked-causal-lm")


class FakeOutput:
    loss = 2.5


class FakeModel:
    def __call__(self, **kwargs):
        return FakeOutput()

    def eval(self):
        return self


class EvaluationTests(unittest.TestCase):
    def test_loss_evaluator_uses_named_denominators(self) -> None:
        plugin = load_toy_plugin()
        fixture = FRAMEWORK_ROOT / "tests" / "fixtures" / "datasets" / "canonical_samples.jsonl"
        streams = open_dataset_streams(
            (DatasetSpec(dataset_id="x", uri=str(fixture), importer="canonical"),),
            source_config=None,
        )
        batches = StatefulBatchStream(
            MixtureStream(streams, seed=0, repeat=False),
            plugin=plugin,
            sample_objective="sft",
            stage_id="eval",
            budget=BatchBudget(max_samples=1),
            packing=False,
        )
        objective = ObjectiveRegistry().get("masked-causal-lm")
        result = LossEvaluator().evaluate(
            EvaluationRequest(
                run_name="eval",
                model_bundle=ModelBundle(FakeModel()),
                batches=batches,
                objective=objective,
                output_dir=FRAMEWORK_ROOT,
                config={"max_batches": 1},
            )
        )
        self.assertEqual(result.metrics["loss/token_ce"], 2.5)
        self.assertEqual(result.counts["batches"], 1)


if __name__ == "__main__":
    unittest.main()
