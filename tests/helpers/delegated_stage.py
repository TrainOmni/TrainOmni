from __future__ import annotations

import json
import os
from pathlib import Path

request = Path(os.environ["TRAINOMNI_STAGE_REQUEST"])
payload = json.loads(request.read_text(encoding="utf-8"))
result = {
    "status": "succeeded",
    "metrics": {"reward/mean": 0.75, "rollouts": 4},
    "outputs": {
        "checkpoint": {
            "artifact_id": "delegated/grpo/step-1",
            "selector": "last",
            "uri": str((Path.cwd() / "backend-checkpoint").resolve()),
        }
    },
}
(Path.cwd() / "stage-result.json").write_text(
    json.dumps(result), encoding="utf-8"
)
