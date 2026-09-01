import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from trainomni.core.errors import SpecError
from trainomni.modules.export.transformers.config import TransformersExportConfig
from trainomni.modules.export.transformers.module import TransformersExporter
from trainomni.modules.model.models.monolithic.module import MonolithicModel


def test_transformers_export_reloads_in_fresh_process_with_equal_logits(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    wrapped = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=23,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    input_ids = torch.tensor([[1, 4, 8, 2]], dtype=torch.long)
    with torch.inference_mode():
        expected = wrapped(input_ids=input_ids).logits

    destination = tmp_path / "artifact"
    artifact = TransformersExporter(
        TransformersExportConfig(save_processor=False, max_shard_size="1MB")
    ).export(
        model=MonolithicModel(wrapped),
        destination=destination,
        identity={"task_digest": "a" * 64, "run_digest": "b" * 64},
    )

    manifest = json.loads(
        (destination / "trainomni-export.json").read_text(encoding="utf-8")
    )
    assert artifact.kind == "transformers"
    assert artifact.uri == str(destination.resolve())
    assert artifact.digest == manifest["payload_tree_sha256"]
    assert (destination / "model.safetensors").is_file()

    expected_path = tmp_path / "expected.pt"
    input_path = tmp_path / "input.pt"
    output_path = tmp_path / "actual.pt"
    torch.save(expected, expected_path)
    torch.save(input_ids, input_path)
    environment = dict(os.environ)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, torch; "
                "from transformers import AutoModelForCausalLM; "
                "model=AutoModelForCausalLM.from_pretrained(sys.argv[1], "
                "local_files_only=True).eval(); "
                "ids=torch.load(sys.argv[2], weights_only=True); "
                "torch.save(model(input_ids=ids).logits, sys.argv[3])"
            ),
            str(destination),
            str(input_path),
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    actual = torch.load(output_path, weights_only=True)
    assert torch.equal(actual, torch.load(expected_path, weights_only=True))


def test_transformers_export_requires_processor_when_configured(tmp_path: Path) -> None:
    model = MonolithicModel(
        GPT2LMHeadModel(GPT2Config(vocab_size=11, n_embd=8, n_layer=1, n_head=1))
    )
    exporter = TransformersExporter(TransformersExportConfig(save_processor=True))

    with pytest.raises(SpecError, match="exposes no processor"):
        exporter.export(
            model=model,
            destination=tmp_path / "missing-processor",
            identity={"task_digest": "a" * 64},
        )


def test_transformers_export_is_registered_with_monolithic_boundary() -> None:
    descriptor = __import__(
        "trainomni.modules.export.transformers.module", fromlist=["descriptor"]
    ).descriptor()
    assert descriptor.requires.values == frozenset({"model.monolithic"})
    assert descriptor.provides.values == frozenset({"export.transformers"})
    assert AutoModelForCausalLM is not None
