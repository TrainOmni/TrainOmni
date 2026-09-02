from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
from pyarrow import parquet


def test_two_rank_gloo_physical_sharding_fresh_process_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.parquet"
    parquet.write_table(
        pa.Table.from_pylist([{"value": index} for index in range(8)]),
        path,
        row_group_size=2,
    )
    worker = Path(__file__).with_name("_data_sharding_worker.py")
    environment = dict(os.environ)
    source_root = str(Path(__file__).parents[3] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    environment["USE_LIBUV"] = "0"
    for phase in ("capture", "resume"):
        completed = subprocess.run(
            [sys.executable, str(worker), phase, str(tmp_path), str(path)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=90,
        )
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    capture = json.loads((tmp_path / "capture.json").read_text(encoding="utf-8"))
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert set(capture[0]["prefix"]).isdisjoint(capture[1]["prefix"])
    assert [item["actual"] for item in result] == [
        item["expected"] for item in result
    ]
    assert all(item["state"]["emitted"] == 9 for item in result)
