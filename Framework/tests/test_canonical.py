from __future__ import annotations

import copy
import json
import re
import sys
import unittest
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.data import (
    SCHEMA_VERSION,
    SampleValidationError,
    canonical_hash,
    canonical_json,
    load_sample,
    parse_sample,
    validate_sample_dict,
)
from trainomni.data.model import BLOCK_TYPES, OBJECTIVES, ROLES

VALID_DIR = FRAMEWORK_ROOT / "tests" / "fixtures" / "valid"
INVALID_DIR = FRAMEWORK_ROOT / "tests" / "fixtures" / "invalid"


class CanonicalSampleTests(unittest.TestCase):
    def test_all_valid_fixtures_parse(self) -> None:
        fixtures = sorted(VALID_DIR.glob("*.json"))
        self.assertGreaterEqual(len(fixtures), 4)
        for path in fixtures:
            with self.subTest(path=path.name):
                sample = load_sample(path)
                self.assertEqual(sample.schema_version, SCHEMA_VERSION)
                self.assertRegex(canonical_hash(sample), r"^sha256:[0-9a-f]{64}$")
                self.assertEqual(parse_sample(json.loads(canonical_json(sample))), sample)

    def test_invalid_fixtures_report_expected_semantic_codes(self) -> None:
        expected_codes = {
            "bbox_out_of_bounds.json": "bbox.bounds.pixel",
            "dangling_asset.json": "asset.missing_reference",
            "duplicate_asset.json": "asset.duplicate_id",
            "identical_preference.json": "preference.identical",
            "orphan_tool_result.json": "tool_result.orphan",
            "prompt_missing_rollout.json": "rollout.required",
            "sft_no_assistant.json": "sft.assistant_required",
        }
        self.assertEqual(
            set(expected_codes), {path.name for path in INVALID_DIR.glob("*.json")}
        )
        for filename, expected_code in expected_codes.items():
            with self.subTest(filename=filename):
                raw = json.loads((INVALID_DIR / filename).read_text(encoding="utf-8"))
                issues = validate_sample_dict(raw)
                self.assertIn(expected_code, {issue.code for issue in issues})
                with self.assertRaises(SampleValidationError) as caught:
                    parse_sample(raw)
                self.assertIn(expected_code, str(caught.exception))

    def test_hash_is_independent_of_key_order_and_empty_defaults(self) -> None:
        raw = json.loads((VALID_DIR / "sft_grounding.json").read_text(encoding="utf-8"))
        reordered = {key: copy.deepcopy(raw[key]) for key in reversed(raw)}
        reordered["tools"] = []
        self.assertEqual(
            canonical_hash(parse_sample(raw)), canonical_hash(parse_sample(reordered))
        )

    def test_numeric_coordinates_are_normalized(self) -> None:
        raw = json.loads((VALID_DIR / "sft_grounding.json").read_text(encoding="utf-8"))
        floating = copy.deepcopy(raw)
        floating["messages"][1]["content"][1]["xyxy"] = [
            244.0,
            301.0,
            612.0,
            633.0,
        ]
        self.assertEqual(
            canonical_hash(parse_sample(raw)), canonical_hash(parse_sample(floating))
        )

    def test_nested_json_is_frozen_and_detached_from_source(self) -> None:
        raw = json.loads((VALID_DIR / "prompt_only.json").read_text(encoding="utf-8"))
        sample = parse_sample(raw)
        raw["rollout"]["verifiers"][0]["spec"]["absolute_tolerance"] = 9
        self.assertEqual(
            sample.rollout.verifiers[0].spec["absolute_tolerance"], 0.001
        )
        with self.assertRaises(TypeError):
            sample.rollout.verifiers[0].spec["new"] = True

    def test_structural_errors_include_precise_path(self) -> None:
        raw = {
            "schema_version": SCHEMA_VERSION,
            "id": "bad-unknown-field",
            "objective": "sft",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "x", "surprise": 1}],
                }
            ],
        }
        issues = validate_sample_dict(raw)
        self.assertEqual(issues[0].path, "$.messages[0].content[0].surprise")
        self.assertEqual(issues[0].code, "unknown_field")

    def test_paired_tool_call_and_result_are_valid(self) -> None:
        raw = {
            "schema_version": SCHEMA_VERSION,
            "id": "tool-valid",
            "objective": "sft",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Check."}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "name": "inspect",
                            "arguments": {"target": "image"},
                            "call_id": "call-1",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "value": {"ok": True},
                            "call_id": "call-1",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
            ],
        }
        self.assertEqual(validate_sample_dict(raw), ())

    def test_runtime_constants_match_documented_json_schema(self) -> None:
        schema = json.loads(
            (FRAMEWORK_ROOT / "docs" / "design" / "sample-v0.1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertEqual(set(schema["properties"]["objective"]["enum"]), OBJECTIVES)
        self.assertEqual(set(schema["$defs"]["message"]["properties"]["role"]["enum"]), ROLES)
        block_names = {
            definition["properties"]["type"]["const"]
            for name, definition in schema["$defs"].items()
            if name.endswith("_block") and "properties" in definition
        }
        self.assertEqual(block_names, BLOCK_TYPES)

    def test_hash_shape_is_versioned(self) -> None:
        sample = load_sample(VALID_DIR / "cpt_interleaved.json")
        fingerprint = canonical_hash(sample)
        self.assertTrue(fingerprint.startswith("sha256:"))
        self.assertIsNotNone(re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint))


if __name__ == "__main__":
    unittest.main()

