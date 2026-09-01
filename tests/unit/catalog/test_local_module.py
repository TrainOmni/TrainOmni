import sys
from pathlib import Path

import pytest

from trainomni.catalog.local import load_local_descriptor, source_tree_digest
from trainomni.core.errors import RegistryError
from trainomni.specs.task import LocalModuleSpec


def create_module(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / "config.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Config:\n"
        "    scale: float = 1.0\n",
        encoding="utf-8",
    )
    (directory / "module.py").write_text(
        "from trainomni.core.module import ModuleDescriptor, ModuleId\n"
        "from .config import Config\n"
        "def descriptor():\n"
        "    return ModuleDescriptor(\n"
        "        module_id=ModuleId.parse('objective:task/weighted_ce@1'),\n"
        "        config_type=Config,\n"
        "        factory=lambda config, context: (config, context),\n"
        "    )\n",
        encoding="utf-8",
    )
    (directory / "module.toml").write_text(
        "[module]\n"
        'id = "objective:task/weighted_ce@1"\n'
        'entrypoint = "module:descriptor"\n'
        "api_version = 1\n",
        encoding="utf-8",
    )


def test_local_module_is_explicit_hash_pinned_and_does_not_mutate_sys_path(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "modules" / "weighted_ce"
    create_module(directory)
    source = LocalModuleSpec.from_mapping(
        {
            "module": "objective:task/weighted_ce@1",
            "path": "modules/weighted_ce",
            "source_sha256": source_tree_digest(directory),
        },
        field="local_modules[0]",
    )
    before = tuple(sys.path)
    with pytest.raises(RegistryError, match="explicit opt-in"):
        load_local_descriptor(source, task_root=tmp_path, allow_local_code=False)
    descriptor = load_local_descriptor(source, task_root=tmp_path, allow_local_code=True)
    assert str(descriptor.module_id) == "objective:task/weighted_ce@1"
    assert tuple(sys.path) == before


def test_local_module_tamper_fails_before_import(tmp_path: Path) -> None:
    directory = tmp_path / "modules" / "weighted_ce"
    create_module(directory)
    digest = source_tree_digest(directory)
    source = LocalModuleSpec.from_mapping(
        {
            "module": "objective:task/weighted_ce@1",
            "path": "modules/weighted_ce",
            "source_sha256": digest,
        },
        field="local_modules[0]",
    )
    (directory / "config.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="source digest mismatch"):
        load_local_descriptor(source, task_root=tmp_path, allow_local_code=True)
