import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainomni.core.errors import CheckpointError
from trainomni.runtime.execution.process import ProcessContext


def test_build_engine_coordinates_identity_failure_and_closes_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_api = importlib.import_module("trainomni.api.train")
    created = []

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.process = ProcessContext("single", 0, 0, 1, None, False)
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(train_api, "TrainEngine", FakeEngine)
    monkeypatch.setattr(
        train_api,
        "materialize_run_identity",
        lambda **kwargs: (_ for _ in ()).throw(OSError("receipt is read-only")),
    )
    assembly = SimpleNamespace(
        parameter_selection=SimpleNamespace(),
        model=SimpleNamespace(),
        objective=SimpleNamespace(),
        stream=SimpleNamespace(),
        module_lock={"module": "a" * 64},
        reproducible=True,
        provenance_issues=(),
    )
    run = SimpleNamespace(
        checkpoint=SimpleNamespace(directory=tmp_path / "checkpoints")
    )

    with pytest.raises(CheckpointError, match="receipt is read-only"):
        train_api.build_engine(
            task=SimpleNamespace(digest="b" * 64),
            assembly=assembly,
            run=run,
        )

    assert created[0].closed
