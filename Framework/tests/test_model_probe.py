from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainomni.models import (
    ComponentCatalog,
    ComponentRule,
    ModelCapabilities,
    ModelRequirements,
    ProbeError,
    analyze_composite,
    negotiate_capabilities,
    probe_checkpoint,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], int]]
) -> None:
    offset = 0
    header: dict[str, object] = {}
    for name, (dtype, shape, byte_count) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        offset += byte_count
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0" * offset)


def _make_checkpoint(
    root: Path,
    *,
    config: dict[str, object],
    tensors: dict[str, tuple[str, list[int], int]],
    tokenizer: dict[str, object] | None = None,
    processor: dict[str, object] | None = None,
) -> None:
    root.mkdir()
    _write_json(root / "config.json", config)
    if tokenizer is not None:
        _write_json(root / "tokenizer_config.json", tokenizer)
    if processor is not None:
        _write_json(root / "preprocessor_config.json", processor)
    filename = "model.safetensors"
    _write_safetensors(root / filename, tensors)
    _write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {name: filename for name in tensors}},
    )


class ModelProbeTests(unittest.TestCase):
    def test_static_probe_and_composite_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vision_path = root / "vision"
            language_path = root / "language"
            _make_checkpoint(
                vision_path,
                config={
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "text_config": {
                        "hidden_size": 1024,
                        "vocab_size": 248320,
                        "max_position_embeddings": 262144,
                        "dtype": "bfloat16",
                    },
                    "vision_config": {
                        "hidden_size": 768,
                        "out_hidden_size": 1024,
                    },
                },
                tensors={
                    "model.visual.merger.linear_fc2.weight": (
                        "BF16",
                        [1024, 3072],
                        16,
                    ),
                    "model.language_model.embed_tokens.weight": (
                        "BF16",
                        [248320, 1024],
                        16,
                    ),
                },
                processor={"processor_class": "Qwen3VLProcessor"},
            )
            _make_checkpoint(
                language_path,
                config={
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "hidden_size": 1536,
                    "vocab_size": 130560,
                    "max_position_embeddings": 131072,
                    "torch_dtype": "bfloat16",
                },
                tensors={
                    "model.embed_tokens.weight": ("BF16", [130560, 1536], 16),
                    "lm_head.weight": ("BF16", [130560, 1536], 16),
                },
                tokenizer={
                    "added_tokens_decoder": {
                        "130082": {"content": "<unused_token_0>", "special": False}
                    }
                },
            )

            vision = probe_checkpoint(vision_path)
            language = probe_checkpoint(language_path)
            report = analyze_composite(vision, language)

            self.assertEqual(len(vision.tensors), 2)
            self.assertEqual(vision.tensor("model.visual.merger.linear_fc2.weight").shape, (1024, 3072))
            self.assertFalse(vision.remote_code_required)
            self.assertTrue(report.compatible)
            self.assertEqual(report.connector_in_dim, 1024)
            self.assertEqual(report.connector_out_dim, 1536)
            self.assertEqual(report.reserved_placeholder_token, "<unused_token_0>")
            self.assertEqual(report.reserved_placeholder_id, 130082)
            self.assertTrue(any("different vocabularies" in item for item in report.warnings))

    def test_probe_rejects_index_header_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bad"
            _make_checkpoint(
                root,
                config={"architectures": ["Test"], "model_type": "test"},
                tensors={"weight": ("F32", [1], 4)},
            )
            _write_json(
                root / "model.safetensors.index.json",
                {"weight_map": {"different": "model.safetensors"}},
            )
            with self.assertRaisesRegex(ProbeError, "index/header tensor mismatch"):
                probe_checkpoint(root)

    def test_capability_negotiation_reports_all_mismatches(self) -> None:
        capabilities = ModelCapabilities(
            modalities=frozenset({"text", "image"}),
            content_blocks=frozenset({"text", "media"}),
            objectives=frozenset({"sft"}),
            max_media_per_sample=1,
            supports_generation=True,
            attention_backends=frozenset({"sdpa"}),
            parallelism=frozenset({"single", "ddp"}),
        )
        requirements = ModelRequirements(
            modalities=frozenset({"image", "video"}),
            content_blocks=frozenset({"text", "bbox"}),
            objectives=frozenset({"preference"}),
            media_per_sample=2,
            require_packing=True,
            attention_backend="flash_attention_2",
            parallelism="fsdp2",
        )
        report = negotiate_capabilities(requirements, capabilities)
        self.assertFalse(report.compatible)
        self.assertEqual(
            {issue.code for issue in report.issues},
            {
                "capability.modalities",
                "capability.content_blocks",
                "capability.objectives",
                "capability.media_count",
                "capability.packing",
                "capability.attention_backend",
                "capability.parallelism",
            },
        )

    def test_component_catalog_requires_exact_assignment(self) -> None:
        catalog = ComponentCatalog(
            rules=(
                ComponentRule("vision_encoder", ("vision.",)),
                ComponentRule("connector", ("connector.",)),
                ComponentRule("language_model", ("language.",)),
            )
        )
        assignments, issues = catalog.classify(
            ["vision.block.weight", "connector.proj.weight", "language.layer.weight"]
        )
        self.assertEqual(issues, ())
        self.assertEqual(assignments["connector"], ("connector.proj.weight",))

        _, broken = catalog.classify(["vision.block.weight", "unknown.weight"])
        self.assertIn("component.unclassified", {issue.code for issue in broken})
        self.assertIn("component.empty", {issue.code for issue in broken})


if __name__ == "__main__":
    unittest.main()

