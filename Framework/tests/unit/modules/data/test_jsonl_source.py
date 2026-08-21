import hashlib
import json
from pathlib import Path

import pytest

from trainomni.core.context import BuildContext
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleRef
from trainomni.modules.data.sources.jsonl.module import descriptor


def write_jsonl(path: Path) -> str:
    lines = [
        {
            "sample_id": "a",
            "content": [{"kind": "text", "value": "first"}],
        },
        {
            "sample_id": "b",
            "content": [{"kind": "text", "value": "second"}],
        },
    ]
    payload = "".join(json.dumps(line) + "\n" for line in lines).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build_source(task_root: Path, digest: str):
    module = descriptor()
    reference = ModuleRef.from_mapping(
        {
            "module": "data_source:trainomni/jsonl@1",
            "config": {"path": "samples.jsonl", "sha256": digest},
        },
        field_name="data.source",
    )
    return module.build(reference, BuildContext("task", task_root))


def test_jsonl_source_cursor_round_trips_exactly(tmp_path: Path) -> None:
    digest = write_jsonl(tmp_path / "samples.jsonl")
    source = build_source(tmp_path, digest)
    assert source.next_sample().sample_id == "a"
    state = source.state_dict()
    assert source.next_sample().sample_id == "b"

    restored = build_source(tmp_path, digest)
    restored.load_state_dict(state)
    assert restored.next_sample().sample_id == "b"
    assert restored.next_sample().sample_id == "a"
    assert restored.epoch == 1


def test_jsonl_identity_tamper_fails_before_iteration(tmp_path: Path) -> None:
    digest = write_jsonl(tmp_path / "samples.jsonl")
    (tmp_path / "samples.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SpecError, match="digest mismatch"):
        build_source(tmp_path, digest)


def test_jsonl_source_parses_role_aware_multimodal_messages(tmp_path: Path) -> None:
    payload = (
        json.dumps(
            {
                "sample_id": "chat",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"kind": "image", "value": "image.png"},
                            {"kind": "text", "value": "question"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"kind": "text", "value": "answer"}],
                    },
                ],
            }
        )
        + "\n"
    ).encode()
    path = tmp_path / "samples.jsonl"
    path.write_bytes(payload)
    source = build_source(tmp_path, hashlib.sha256(payload).hexdigest())
    sample = source.next_sample()
    assert sample.content == ()
    assert tuple(message.role for message in sample.messages) == ("user", "assistant")
    assert tuple(block.kind for block in sample.messages[0].content) == (
        "image",
        "text",
    )
