from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_two_rank_global_loss_and_primary_failures_are_coordinated(
    tmp_path: Path,
) -> None:
    worker = Path(__file__).with_name("_ddp_correctness_worker.py")
    result = tmp_path / "result.json"
    environment = dict(os.environ)
    source_root = str(Path(__file__).parents[3] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    environment["USE_LIBUV"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            str(result),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["weights"] == pytest.approx(
        [payload["expected_weight"], payload["expected_weight"]],
        rel=1e-7,
        abs=1e-7,
    )
    assert payload["loss"] == pytest.approx(
        payload["expected_loss"], rel=1e-7, abs=1e-7
    )
    assert len(set(payload["checkpoint_errors"])) == 1
    assert "refusing to overwrite checkpoint" in payload["checkpoint_errors"][0]
    assert len(set(payload["materialize_errors"])) == 1
    assert "identity receipt denied" in payload["materialize_errors"][0]
