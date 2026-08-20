#!/usr/bin/env python3
"""Deterministic registry for a project handoff coordinator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
VALID_STATUSES = {
    "registered",
    "active",
    "idle",
    "needs_input",
    "blocked",
    "complete",
    "stale",
}
HANDOFF_ROOT = Path(__file__).resolve().parents[3]


class RegistryError(RuntimeError):
    """A user-correctable registry error."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_paths(state_dir: str | None) -> tuple[Path, Path]:
    root = Path(state_dir).resolve() if state_dir else HANDOFF_ROOT / "state"
    return root / "registry.json", root / "events.jsonl"


def require_text(value: str, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise RegistryError(f"{field} must not be blank")
    return cleaned


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegistryError(f"registry is not initialized: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry: {exc}") from exc
    validate_registry(data)
    return data


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_registry(data: dict[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError("unsupported or missing schema_version")
    coordinator = data.get("coordinator")
    if not isinstance(coordinator, dict):
        raise RegistryError("coordinator must be an object")
    for field in ("thread_id", "title", "host_id"):
        if not isinstance(coordinator.get(field), str) or not coordinator[field].strip():
            raise RegistryError(f"coordinator.{field} must be non-empty")
    if not isinstance(data.get("project_root"), str) or not data["project_root"].strip():
        raise RegistryError("project_root must be non-empty")
    members = data.get("members")
    if not isinstance(members, dict):
        raise RegistryError("members must be an object")

    thread_ids: set[str] = set()
    for role, member in members.items():
        if not isinstance(role, str) or not role.strip():
            raise RegistryError("member roles must be non-empty strings")
        if not isinstance(member, dict):
            raise RegistryError(f"member {role!r} must be an object")
        for field in ("thread_id", "scope", "status", "registered_at", "updated_at"):
            if not isinstance(member.get(field), str) or not member[field].strip():
                raise RegistryError(f"member {role!r}.{field} must be non-empty")
        if member["status"] not in VALID_STATUSES:
            raise RegistryError(f"member {role!r} has invalid status")
        if member["thread_id"] in thread_ids:
            raise RegistryError(f"thread_id is registered more than once: {member['thread_id']}")
        thread_ids.add(member["thread_id"])


def make_event(event_type: str, **fields: Any) -> dict[str, Any]:
    event = {"at": now(), "type": require_text(event_type, "type")}
    event.update({key: value for key, value in fields.items() if value is not None})
    return event


def output(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def command_init(args: argparse.Namespace) -> None:
    registry_path, events_path = state_paths(args.state_dir)
    if registry_path.exists() and not args.force:
        existing = load_registry(registry_path)
        desired = {
            "thread_id": require_text(args.thread_id, "thread_id"),
            "title": require_text(args.title, "title"),
            "host_id": require_text(args.host_id, "host_id"),
        }
        if (
            existing["project_root"] == str(Path(args.project_root).resolve())
            and existing["coordinator"] == desired
        ):
            output(existing)
            return
        raise RegistryError("registry already exists with different coordinator metadata")

    timestamp = now()
    data = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(Path(require_text(args.project_root, "project_root")).resolve()),
        "coordinator": {
            "thread_id": require_text(args.thread_id, "thread_id"),
            "title": require_text(args.title, "title"),
            "host_id": require_text(args.host_id, "host_id"),
        },
        "members": {},
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    validate_registry(data)
    atomic_write(registry_path, data)
    append_event(events_path, make_event("registry_initialized"))
    output(data)


def command_register(args: argparse.Namespace) -> None:
    registry_path, events_path = state_paths(args.state_dir)
    data = load_registry(registry_path)
    role = require_text(args.role, "role")
    thread_id = require_text(args.thread_id, "thread_id")
    scope = require_text(args.scope, "scope")
    existing = data["members"].get(role)

    if existing and existing["thread_id"] == thread_id and existing["scope"] == scope:
        output({"result": "already_registered", "role": role, **existing})
        return
    if existing and not args.replace:
        raise RegistryError(f"role is already registered: {role}")

    conflicting_role = next(
        (
            member_role
            for member_role, member in data["members"].items()
            if member["thread_id"] == thread_id and member_role != role
        ),
        None,
    )
    if conflicting_role and not args.replace:
        raise RegistryError(f"thread_id is already registered as role: {conflicting_role}")

    replaced: list[str] = []
    if args.replace:
        if existing:
            replaced.append(role)
            del data["members"][role]
        if conflicting_role:
            replaced.append(conflicting_role)
            del data["members"][conflicting_role]

    timestamp = now()
    data["members"][role] = {
        "thread_id": thread_id,
        "scope": scope,
        "status": "registered",
        "registered_at": timestamp,
        "updated_at": timestamp,
    }
    data["updated_at"] = timestamp
    validate_registry(data)
    atomic_write(registry_path, data)
    append_event(
        events_path,
        make_event(
            "registered",
            role=role,
            thread_id=thread_id,
            scope=scope,
            replaced=replaced or None,
        ),
    )
    output({"result": "registered", "role": role, **data["members"][role]})


def command_unregister(args: argparse.Namespace) -> None:
    registry_path, events_path = state_paths(args.state_dir)
    data = load_registry(registry_path)
    role = require_text(args.role, "role")
    reason = require_text(args.reason, "reason")
    member = data["members"].get(role)
    if member is None:
        raise RegistryError(f"role is not registered: {role}")
    del data["members"][role]
    data["updated_at"] = now()
    validate_registry(data)
    atomic_write(registry_path, data)
    append_event(
        events_path,
        make_event(
            "unregistered",
            role=role,
            thread_id=member["thread_id"],
            reason=reason,
        ),
    )
    output({"result": "unregistered", "role": role, "thread_id": member["thread_id"]})


def command_status(args: argparse.Namespace) -> None:
    registry_path, events_path = state_paths(args.state_dir)
    data = load_registry(registry_path)
    role = require_text(args.role, "role")
    member = data["members"].get(role)
    if member is None:
        raise RegistryError(f"role is not registered: {role}")
    member["status"] = args.value
    member["updated_at"] = now()
    data["updated_at"] = member["updated_at"]
    validate_registry(data)
    atomic_write(registry_path, data)
    append_event(events_path, make_event("status", role=role, status=args.value))
    output({"result": "status_updated", "role": role, **member})


def command_event(args: argparse.Namespace) -> None:
    registry_path, events_path = state_paths(args.state_dir)
    data = load_registry(registry_path)
    from_role = require_text(args.from_role, "from_role")
    to_role = require_text(args.to_role, "to_role") if args.to_role else None
    allowed = set(data["members"]) | {"handoff"}
    if from_role not in allowed:
        raise RegistryError(f"source role is not registered: {from_role}")
    if to_role and to_role not in allowed:
        raise RegistryError(f"target role is not registered: {to_role}")
    event = make_event(
        args.type,
        from_role=from_role,
        to_role=to_role,
        request_id=require_text(args.request_id, "request_id") if args.request_id else None,
        summary=require_text(args.summary, "summary"),
    )
    append_event(events_path, event)
    output({"result": "event_logged", "event": event})


def command_migrate_root(args: argparse.Namespace) -> None:
    registry_path, events_path = state_paths(args.state_dir)
    data = load_registry(registry_path)
    old_root = os.path.abspath(os.path.normpath(require_text(args.from_root, "from_root")))
    new_root = os.path.abspath(os.path.normpath(require_text(args.to_root, "to_root")))
    current_root = os.path.abspath(os.path.normpath(data["project_root"]))

    if os.path.normcase(current_root) == os.path.normcase(new_root):
        output({"result": "already_migrated", "from_root": old_root, "to_root": new_root})
        return
    if os.path.normcase(current_root) != os.path.normcase(old_root):
        raise RegistryError(
            f"project_root does not match from_root: {data['project_root']}"
        )

    def replace_root(value: str) -> str:
        if os.name == "nt":
            start = value.lower().find(old_root.lower())
        else:
            start = value.find(old_root)
        if start < 0:
            return value
        return value[:start] + new_root + value[start + len(old_root) :]

    timestamp = now()
    data["project_root"] = new_root
    updated_roles: list[str] = []
    for role, member in data["members"].items():
        migrated_scope = replace_root(member["scope"])
        if migrated_scope != member["scope"]:
            member["scope"] = migrated_scope
            member["updated_at"] = timestamp
            updated_roles.append(role)
    data["updated_at"] = timestamp
    validate_registry(data)
    atomic_write(registry_path, data)
    append_event(
        events_path,
        make_event(
            "root_migrated",
            from_root=old_root,
            to_root=new_root,
            updated_roles=updated_roles,
        ),
    )
    output(
        {
            "result": "root_migrated",
            "from_root": old_root,
            "to_root": new_root,
            "updated_roles": updated_roles,
        }
    )


def command_list(args: argparse.Namespace) -> None:
    registry_path, _ = state_paths(args.state_dir)
    output(load_registry(registry_path))


def command_validate(args: argparse.Namespace) -> None:
    registry_path, events_path = state_paths(args.state_dir)
    data = load_registry(registry_path)
    event_count = 0
    if events_path.exists():
        for line_number, line in enumerate(
            events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegistryError(f"invalid event JSON at line {line_number}") from exc
            if not isinstance(event, dict) or not event.get("type") or not event.get("at"):
                raise RegistryError(f"invalid event record at line {line_number}")
            event_count += 1
    output(
        {
            "result": "valid",
            "registry": str(registry_path),
            "members": len(data["members"]),
            "events": event_count,
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--state-dir",
        help="Override the state directory for isolated tests only.",
    )
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("--project-root", required=True)
    init.add_argument("--thread-id", required=True)
    init.add_argument("--title", default="handoff")
    init.add_argument("--host-id", default="local")
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=command_init)

    register = commands.add_parser("register")
    register.add_argument("--role", required=True)
    register.add_argument("--thread-id", required=True)
    register.add_argument("--scope", required=True)
    register.add_argument("--replace", action="store_true")
    register.set_defaults(handler=command_register)

    unregister = commands.add_parser("unregister")
    unregister.add_argument("--role", required=True)
    unregister.add_argument("--reason", required=True)
    unregister.set_defaults(handler=command_unregister)

    status = commands.add_parser("status")
    status.add_argument("--role", required=True)
    status.add_argument("--value", required=True, choices=sorted(VALID_STATUSES))
    status.set_defaults(handler=command_status)

    event = commands.add_parser("event")
    event.add_argument("--type", required=True)
    event.add_argument("--from-role", required=True)
    event.add_argument("--to-role")
    event.add_argument("--request-id")
    event.add_argument("--summary", required=True)
    event.set_defaults(handler=command_event)

    migrate_root = commands.add_parser("migrate-root")
    migrate_root.add_argument("--from-root", required=True)
    migrate_root.add_argument("--to-root", required=True)
    migrate_root.set_defaults(handler=command_migrate_root)

    listing = commands.add_parser("list")
    listing.set_defaults(handler=command_list)

    validate = commands.add_parser("validate")
    validate.set_defaults(handler=command_validate)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
        return 0
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
