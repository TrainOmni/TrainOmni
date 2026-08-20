from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.checkpoint import (
    CheckpointError,
    LocalCheckpointManager,
    PythonRandomState,
    ScalarState,
    StateRegistry,
    StateRegistryError,
)
from trainomni.config import load_run_spec
from trainomni.data import DatasetStream

CONFIG = FRAMEWORK_ROOT / "configs" / "examples" / "toy_data_inspect.yaml"


class CheckpointTests(unittest.TestCase):
    def test_exact_local_resume_restores_reader_step_and_rng(self) -> None:
        spec = load_run_spec(CONFIG).stage.data.datasets[0]
        stream = DatasetStream(spec, base_dir=CONFIG.parent)
        step = ScalarState(0)
        registry = StateRegistry()
        registry.register("data", stream)
        registry.register("step", step)
        registry.register("python_rng", PythonRandomState())

        first = next(iter(stream)).sample.id
        step.value = 1
        random.seed(12345)
        expected_random = random.random()
        random.seed(12345)

        with tempfile.TemporaryDirectory() as temporary:
            manager = LocalCheckpointManager(temporary)
            manager.save(
                "step-000001", registry, metadata={"run_fingerprint": "abc"}
            )
            second_before_resume = next(iter(stream)).sample.id
            step.value = 99
            random.random()

            metadata = manager.load(
                "step-000001", registry, trusted=True, strict=True
            )
            second_after_resume = next(iter(stream)).sample.id
            resumed_random = random.random()

            self.assertEqual(manager.list_complete(), ("step-000001",))
        self.assertEqual(first, "inspect-sft-001")
        self.assertEqual(second_before_resume, second_after_resume)
        self.assertEqual(step.value, 1)
        self.assertEqual(resumed_random, expected_random)
        self.assertEqual(metadata["run_fingerprint"], "abc")

    def test_untrusted_or_corrupt_checkpoint_is_rejected(self) -> None:
        registry = StateRegistry()
        registry.register("step", ScalarState(1))
        with tempfile.TemporaryDirectory() as temporary:
            manager = LocalCheckpointManager(temporary)
            path = manager.save("safe", registry)
            with self.assertRaisesRegex(CheckpointError, "trusted=True"):
                manager.load("safe", registry, trusted=False)
            state_path = path / "state.pkl"
            state_path.write_bytes(state_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(CheckpointError, "size mismatch"):
                manager.load("safe", registry, trusted=True)

    def test_state_registry_is_strict_and_checkpoint_names_cannot_escape(self) -> None:
        registry = StateRegistry()
        registry.register("step", ScalarState(1))
        with self.assertRaisesRegex(StateRegistryError, "already registered"):
            registry.register("step", ScalarState(2))
        with tempfile.TemporaryDirectory() as temporary:
            manager = LocalCheckpointManager(temporary)
            with self.assertRaisesRegex(CheckpointError, "invalid checkpoint name"):
                manager.save("../escape", registry)


if __name__ == "__main__":
    unittest.main()
