"""Explicit, hash-pinned task-local module loading.

This is namespace isolation and provenance enforcement, not a security sandbox.
Local code is only executed when the caller explicitly opts in.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import sys
import tomllib
from pathlib import Path

from trainomni.core.errors import RegistryError
from trainomni.core.module import MODULE_API_VERSION, ModuleDescriptor, ModuleId
from trainomni.core.registry import ModuleRegistry
from trainomni.specs.task import LocalModuleSpec, TaskSpec

_ENTRYPOINT = re.compile(
    r"^(?P<module>[A-Za-z_][A-Za-z0-9_.]*):(?P<attribute>[A-Za-z_][A-Za-z0-9_]*)$"
)


def source_tree_digest(directory: Path) -> str:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise RegistryError(f"local module directory does not exist: {root}")
    digest = hashlib.sha256()
    files = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise RegistryError(f"local module source must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise RegistryError(f"local module source is empty: {root}")
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_local_descriptor(
    source: LocalModuleSpec,
    *,
    task_root: Path,
    allow_local_code: bool,
) -> ModuleDescriptor:
    if not allow_local_code:
        raise RegistryError(
            f"task requests local code for {source.module_id}; explicit opt-in is required"
        )
    root = Path(task_root).resolve()
    directory = (root / source.path).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise RegistryError(f"local module escapes the task root: {source.path}") from exc
    actual_digest = source_tree_digest(directory)
    if actual_digest != source.source_sha256:
        raise RegistryError(
            f"local module source digest mismatch for {source.module_id}: "
            f"expected {source.source_sha256}, got {actual_digest}"
        )
    manifest_path = directory / "module.toml"
    package_init = directory / "__init__.py"
    if not manifest_path.is_file() or not package_init.is_file():
        raise RegistryError(
            f"local module {source.module_id} requires module.toml and __init__.py"
        )
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RegistryError(f"invalid local module manifest: {exc}") from exc
    module_table = manifest.get("module")
    if not isinstance(module_table, dict):
        raise RegistryError("local module manifest requires a [module] table")
    allowed = {"id", "entrypoint", "api_version"}
    unknown = sorted(set(module_table) - allowed)
    missing = sorted({"id", "entrypoint", "api_version"} - set(module_table))
    if unknown or missing:
        raise RegistryError(
            f"invalid local module manifest keys; missing={missing}, unknown={unknown}"
        )
    manifest_id = ModuleId.parse(str(module_table["id"]))
    if manifest_id != source.module_id:
        raise RegistryError(
            f"local module id mismatch: task={source.module_id}, manifest={manifest_id}"
        )
    if module_table["api_version"] != MODULE_API_VERSION:
        raise RegistryError(
            f"local module API mismatch: {module_table['api_version']!r}"
        )
    entrypoint = str(module_table["entrypoint"])
    match = _ENTRYPOINT.fullmatch(entrypoint)
    if match is None:
        raise RegistryError("module.entrypoint must be '<relative.module>:<attribute>'")

    package_name = f"_trainomni_local_{actual_digest}"
    loaded_before = {
        name for name in sys.modules if name == package_name or name.startswith(package_name + ".")
    }
    try:
        package = sys.modules.get(package_name)
        if package is None:
            package_spec = importlib.util.spec_from_file_location(
                package_name,
                package_init,
                submodule_search_locations=[str(directory)],
            )
            if package_spec is None or package_spec.loader is None:
                raise RegistryError(f"cannot create import spec for {directory}")
            package = importlib.util.module_from_spec(package_spec)
            sys.modules[package_name] = package
            package_spec.loader.exec_module(package)
        imported = importlib.import_module(f"{package_name}.{match.group('module')}")
        factory = getattr(imported, match.group("attribute"))
        descriptor = factory()
    except RegistryError:
        for name in tuple(sys.modules):
            if (
                name not in loaded_before
                and (name == package_name or name.startswith(package_name + "."))
            ):
                sys.modules.pop(name, None)
        raise
    except Exception as exc:
        for name in tuple(sys.modules):
            if (
                name not in loaded_before
                and (name == package_name or name.startswith(package_name + "."))
            ):
                sys.modules.pop(name, None)
        raise RegistryError(f"cannot load local module {source.module_id}: {exc}") from exc
    if not isinstance(descriptor, ModuleDescriptor):
        raise RegistryError(f"{entrypoint} did not return ModuleDescriptor")
    if descriptor.module_id != source.module_id:
        raise RegistryError(
            f"local descriptor id mismatch: task={source.module_id}, "
            f"descriptor={descriptor.module_id}"
        )
    return descriptor


def registry_for_task(
    base: ModuleRegistry,
    task: TaskSpec,
    *,
    task_root: Path,
    allow_local_code: bool = False,
) -> ModuleRegistry:
    registry = ModuleRegistry(base.descriptors())
    for source in task.local_modules:
        registry.register(
            load_local_descriptor(
                source, task_root=task_root, allow_local_code=allow_local_code
            )
        )
    return registry
