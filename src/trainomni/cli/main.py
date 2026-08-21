"""Argument parser for the thin TrainOmni command boundary."""

from __future__ import annotations

import argparse
import sys

from trainomni.core.errors import TrainOmniError

from .commands import (
    command_evaluate,
    command_export,
    command_inspect,
    command_module_digest,
    command_train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainomni")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--task", required=True)
    train_parser.add_argument("--run", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--stop-after-steps", type=int)
    train_parser.add_argument("--allow-local-code", action="store_true")
    train_parser.set_defaults(handler=command_train)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--task", required=True)
    inspect_parser.add_argument("--allow-local-code", action="store_true")
    inspect_parser.set_defaults(handler=command_inspect)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--task", required=True)
    evaluate_parser.add_argument("--run", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--batches", required=True, type=int)
    evaluate_parser.add_argument("--allow-local-code", action="store_true")
    evaluate_parser.set_defaults(handler=command_evaluate)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--task", required=True)
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--exporter")
    export_parser.add_argument("--destination")
    export_parser.add_argument("--allow-local-code", action="store_true")
    export_parser.set_defaults(handler=command_export)

    digest_parser = subparsers.add_parser("module-digest")
    digest_parser.add_argument("directory")
    digest_parser.set_defaults(handler=command_module_digest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except TrainOmniError as exc:
        print(f"trainomni: {exc}", file=sys.stderr)
        return 2
