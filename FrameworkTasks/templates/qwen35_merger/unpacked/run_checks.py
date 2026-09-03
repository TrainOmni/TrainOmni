"""Run documented commands serially in fresh processes, keeping every log.

Use a new task copy for another full run; never delete an existing output.
The same Python interpreter is used for preparation, workers and training.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from prepare import immutable_write

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "evidence" / "runtime" / "checks"
    if output.exists():
        raise SystemExit("checks directory already exists; use a new task copy for a new verification group")
    output.mkdir(parents=True)
    commands = [["prepare.py"], ["-m", "trainomni", "inspect", "--task", "task.yaml", "--allow-local-code"]]
    for profile in ("baseline", "worker1", "worker2"):
        commands.append(["-m", "trainomni", "train", "--task", "task.yaml", "--run",
                         f"runs/{profile}.yaml", "--allow-local-code"])
    commands.append(["verify_model.py"])
    options = json.loads((ROOT / "options.json").read_text())
    if options["packing"]:
        commands.append(["verify_model.py", "--precision", "fp32"])
    else:
        checkpoint = "outputs/baseline_001/checkpoints/step-00000002"
        common = ["--task", "task.yaml", "--run", "runs/baseline.yaml",
                  "--checkpoint", checkpoint, "--allow-local-code"]
        commands += [
            ["-m", "trainomni", "evaluate", *common, "--batches", "2"],
            ["-m", "trainomni", "export", *common, "--destination", "outputs/export_001"],
            ["verify_export.py"],
        ]
    records = []
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for index, args in enumerate(commands):
        print("CHECK", index, " ".join(args), flush=True)
        completed = subprocess.run([sys.executable, "-B", *args], cwd=ROOT,
                                   env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=600)
        immutable_write(output / f"{index:02d}.log", completed.stdout)
        record = {"args": args, "returncode": completed.returncode, "log": f"{index:02d}.log"}
        records.append(record)
        if completed.returncode:
            immutable_write(output / "failed.json", json.dumps(records, indent=2).encode())
            print(completed.stdout.decode("utf8", errors="replace"), flush=True)
            raise SystemExit(completed.returncode)
    immutable_write(output / "passed.json", json.dumps(records, indent=2).encode())
    print(json.dumps({"task": ROOT.name, "commands_passed": len(records)}))


if __name__ == "__main__":
    main()
