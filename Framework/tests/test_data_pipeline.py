from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.cli import main as cli_main
from trainomni.config import load_run_spec
from trainomni.contracts import BatchBudget, CostVector
from trainomni.data import (
    BatchPlanningError,
    DataReadError,
    DatasetStream,
    GreedyBatchPlanner,
    JsonlReader,
    ParquetReader,
    TarJsonReader,
    inspect_imported_sample,
)
from trainomni.models import EncodedSample

DATASET = FRAMEWORK_ROOT / "tests" / "fixtures" / "datasets" / "canonical_samples.jsonl"
CONFIG = FRAMEWORK_ROOT / "configs" / "examples" / "toy_data_inspect.yaml"
PLUGIN = FRAMEWORK_ROOT / "tests" / "plugins" / "toy_vlm_plugin.py"
PLUGIN_SPEC = f"{PLUGIN}:PLUGIN"


class DataPipelineTests(unittest.TestCase):
    def test_explicit_data_plugin_registers_importer_without_core_edit(self) -> None:
        data_plugin = FRAMEWORK_ROOT / "tests" / "plugins" / "data_alias_plugin.py"
        config = yaml.safe_load((FRAMEWORK_ROOT / "configs" / "examples" / "toy_data_inspect.yaml").read_text(encoding="utf-8"))
        config["stage"]["data"]["datasets"][0]["importer"] = "canonical-alias"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            # Keep the dataset path absolute because the temporary recipe moved.
            config["stage"]["data"]["datasets"][0]["uri"] = str(
                FRAMEWORK_ROOT / "tests" / "fixtures" / "datasets" / "canonical_samples.jsonl"
            )
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--plugin",
                        PLUGIN_SPEC,
                        "--data-plugin",
                        f"{data_plugin}:PLUGIN",
                        "--json",
                        "inspect",
                        "data",
                        str(path),
                        "--samples",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0, stdout.getvalue())
            self.assertEqual(
                json.loads(stdout.getvalue())["samples"][0]["trace"]["importer_id"],
                "canonical-alias",
            )
    def test_multi_budget_planner_splits_and_rejects_oversized_sample(self) -> None:
        planner = GreedyBatchPlanner(
            BatchBudget(max_samples=2, max_text_tokens=6, max_pixels=100)
        )
        encoded = [
            EncodedSample(
                sample_id=f"s{index}",
                model_inputs={"input_ids": [index]},
                cost=CostVector(text_tokens=3, pixels=40),
            )
            for index in range(3)
        ]
        plans = planner.plan(encoded)
        self.assertEqual([len(plan.items) for plan in plans], [2, 1])
        self.assertEqual(plans[0].total_cost.text_tokens, 6)

        oversized = EncodedSample(
            sample_id="oversized",
            model_inputs={"input_ids": [0]},
            cost=CostVector(text_tokens=7),
        )
        with self.assertRaisesRegex(BatchPlanningError, "text_tokens"):
            planner.plan([oversized])

    def test_jsonl_reader_exact_state_resumes_at_next_record(self) -> None:
        reader = JsonlReader(DATASET)
        iterator = iter(reader)
        first = next(iterator)
        state = reader.state_dict()
        self.assertEqual(first.record_index, 0)
        self.assertEqual(state["next_index"], 1)

        resumed = JsonlReader(DATASET)
        resumed.load_state_dict(state)
        second = next(iter(resumed))
        self.assertEqual(second.record_index, 1)
        self.assertNotEqual(first.value["id"], second.value["id"])

    def test_tar_json_reader_exact_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.tar"
            with tarfile.open(path, "w") as archive:
                for index in range(2):
                    payload = json.dumps({"id": f"tar-{index}"}).encode("utf-8")
                    info = tarfile.TarInfo(f"{index}.json")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            reader = TarJsonReader(path)
            self.assertEqual(next(reader).value["id"], "tar-0")
            state = reader.state_dict()
            restored = TarJsonReader(path)
            restored.load_state_dict(state)
            self.assertEqual(next(restored).value["id"], "tar-1")

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None,
        "optional pyarrow is not installed",
    )
    def test_parquet_reader_projection_and_exact_state(self) -> None:
        import pyarrow as arrow
        from pyarrow import parquet

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.parquet"
            table = arrow.table(
                {"id": ["parquet-0", "parquet-1", "parquet-2"], "drop": [1, 2, 3]}
            )
            parquet.write_table(table, path, row_group_size=2)
            reader = ParquetReader(path, columns=("id",))
            self.assertEqual(next(reader).value, {"id": "parquet-0"})
            state = reader.state_dict()
            restored = ParquetReader(path, columns=("id",))
            restored.load_state_dict(state)
            self.assertEqual(
                [record.value["id"] for record in restored],
                ["parquet-1", "parquet-2"],
            )

    def test_reader_rejects_mutated_dataset_state(self) -> None:
        state = JsonlReader(DATASET).state_dict()
        state["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(DataReadError, "fingerprint mismatch"):
            JsonlReader(DATASET).load_state_dict(state)

    def test_dataset_stream_imports_and_traces_canonical_sample(self) -> None:
        spec = load_run_spec(CONFIG).stage.data.datasets[0]
        stream = DatasetStream(spec, base_dir=CONFIG.parent)
        imported = next(iter(stream))
        inspected = inspect_imported_sample(imported, include_canonical=True)
        self.assertEqual(imported.sample.id, "inspect-sft-001")
        self.assertEqual(imported.trace.dataset_id, "inspect_fixture")
        self.assertTrue(imported.trace.sample_hash.startswith("sha256:"))
        self.assertEqual(len(imported.trace.sample_hash.removeprefix("sha256:")), 64)
        self.assertEqual(inspected["summary"]["assets"][0]["modality"], "image")
        self.assertEqual(inspected["canonical"]["id"], "inspect-sft-001")

    def test_invalid_jsonl_reports_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
            reader = JsonlReader(path)
            iterator = iter(reader)
            next(iterator)
            with self.assertRaisesRegex(DataReadError, r":2"):
                next(iterator)

    def test_cli_inspect_data_reads_real_records(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--plugin",
                    PLUGIN_SPEC,
                    "--json",
                    "inspect",
                    "data",
                    str(CONFIG),
                    "--samples",
                    "2",
                    "--include-canonical",
                ]
            )
        self.assertEqual(exit_code, 0, stdout.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["returned_samples"], 2)
        self.assertEqual(
            [item["summary"]["sample_id"] for item in payload["samples"]],
            ["inspect-sft-001", "inspect-sft-002"],
        )
        self.assertIn("canonical", payload["samples"][0])

    def test_cli_inspect_batch_encodes_plans_and_collates(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--plugin",
                    PLUGIN_SPEC,
                    "--json",
                    "inspect",
                    "batch",
                    str(CONFIG),
                    "--samples",
                    "2",
                ]
            )
        self.assertEqual(exit_code, 0, stdout.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(len(payload["encoded"]), 2)
        self.assertEqual(len(payload["batches"]), 1)
        self.assertEqual(
            payload["batches"][0]["sample_ids"],
            ["inspect-sft-001", "inspect-sft-002"],
        )
        self.assertFalse(payload["will_execute_training"])


if __name__ == "__main__":
    unittest.main()
