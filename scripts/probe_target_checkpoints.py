"""Print a dependency-free static compatibility report for two checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.models.probe import (
    analyze_composite,
    probe_checkpoint,
    probe_to_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect configs and SafeTensors headers without loading weights."
    )
    parser.add_argument("--vision", required=True, help="Vision/VLM checkpoint path")
    parser.add_argument("--language", required=True, help="Language checkpoint path")
    args = parser.parse_args()

    vision = probe_checkpoint(args.vision)
    language = probe_checkpoint(args.language)
    compatibility = analyze_composite(vision, language)
    report = {
        "vision": probe_to_dict(vision),
        "language": probe_to_dict(language),
        "composite": {
            "compatible": compatibility.compatible,
            "vision_output_dim": compatibility.vision_output_dim,
            "language_hidden_dim": compatibility.language_hidden_dim,
            "connector_in_dim": compatibility.connector_in_dim,
            "connector_out_dim": compatibility.connector_out_dim,
            "source_vocab_size": compatibility.source_vocab_size,
            "target_vocab_size": compatibility.target_vocab_size,
            "reserved_placeholder_token": compatibility.reserved_placeholder_token,
            "reserved_placeholder_id": compatibility.reserved_placeholder_id,
            "issues": list(compatibility.issues),
            "warnings": list(compatibility.warnings),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if compatibility.compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())

