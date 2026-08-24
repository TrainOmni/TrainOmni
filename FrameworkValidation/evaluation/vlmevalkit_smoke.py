from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

VALIDATION_ROOT = Path(__file__).resolve().parent
VLMEVALKIT_ROOT = Path(
    os.environ.get(
        "VLMEVALKIT_ROOT",
        r"D:\Codex\TrainOmniTemp\framework-upstream-references-20260821\upstreams\VLMEvalKit",
    )
).resolve()
CONFIG_PATH = VALIDATION_ROOT / "vlmevalkit-smoke-config.json"
RUN_ROOT = VALIDATION_ROOT / "runs"
LMU_DATA_ROOT = VALIDATION_ROOT / "lmudata"


def _load_upstream_runner():
    run_path = VLMEVALKIT_ROOT / "run.py"
    if not run_path.is_file():
        raise RuntimeError(f"VLMEvalKit run.py is missing: {run_path}")
    spec = importlib.util.spec_from_file_location("trainomni_vlmevalkit_run", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load VLMEvalKit runner: {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_single_new_run(before: set[Path]) -> Path:
    candidates = {
        path.parent
        for path in RUN_ROOT.glob("TrainOmniSmoke/*/status.json")
        if path.parent not in before
    }
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one new smoke run, found: {sorted(candidates)}")
    return candidates.pop()


def main() -> int:
    os.environ["LMUData"] = str(LMU_DATA_ROOT)
    os.environ.setdefault("PRED_FORMAT", "tsv")

    import vlmeval.vlm
    from vlmeval.vlm.base import BaseModel

    class TrainOmniSmokeModel(BaseModel):
        calls = 0

        def generate_inner(self, message, dataset=None):
            image_items = [item for item in message if item["type"] == "image"]
            text_items = [item for item in message if item["type"] == "text"]
            if len(image_items) != 1 or len(text_items) != 1:
                raise RuntimeError(f"Unexpected multimodal message: {message!r}")
            if not Path(image_items[0]["value"]).is_file():
                raise RuntimeError(f"Image was not materialized: {image_items[0]['value']}")
            if "Options:" not in text_items[0]["value"]:
                raise RuntimeError(f"MCQ prompt was not constructed: {text_items[0]['value']!r}")
            type(self).calls += 1
            return "A"

    vlmeval.vlm.TrainOmniSmokeModel = TrainOmniSmokeModel

    before = {
        path.parent for path in RUN_ROOT.glob("TrainOmniSmoke/*/status.json")
    }
    upstream_run = _load_upstream_runner()
    original_argv = sys.argv
    try:
        sys.argv = [
            str(VLMEVALKIT_ROOT / "run.py"),
            "--config",
            str(CONFIG_PATH),
            "--work-dir",
            str(RUN_ROOT),
            "--mode",
            "all",
            "--judge",
            "exact_matching",
            "--judge-api-nproc",
            "1",
            "--debug",
        ]
        upstream_run.main()
    finally:
        sys.argv = original_argv

    run_dir = _find_single_new_run(before)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    dataset_status = status.get("datasets", {}).get("TrainOmniSmoke", {})
    prediction_files = list(run_dir.glob("TrainOmniSmoke_TrainOmniSmoke.*"))
    score_files = list(run_dir.glob("TrainOmniSmoke_TrainOmniSmoke*_acc.csv"))

    if TrainOmniSmokeModel.calls != 2:
        raise RuntimeError(f"Expected two model calls, observed {TrainOmniSmokeModel.calls}")
    if dataset_status.get("status") != "done" or dataset_status.get("error_message"):
        raise RuntimeError(f"VLMEvalKit reported a failed dataset status: {dataset_status}")
    if dataset_status.get("metrics", {}).get("split=none|Overall") != 1.0:
        raise RuntimeError(f"Unexpected VLMEvalKit metrics: {dataset_status.get('metrics')}")
    if not prediction_files:
        raise RuntimeError(f"VLMEvalKit prediction artifact is missing in {run_dir}")
    if not score_files:
        raise RuntimeError(f"VLMEvalKit score artifact is missing in {run_dir}")

    receipt = {
        "status": "passed",
        "upstream_root": str(VLMEVALKIT_ROOT),
        "run_dir": str(run_dir),
        "model_calls": TrainOmniSmokeModel.calls,
        "prediction_files": [str(path) for path in prediction_files],
        "score_files": [str(path) for path in score_files],
        "vlmevalkit_status": status,
    }
    receipt_path = run_dir / "trainomni-smoke-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
