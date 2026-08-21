"""CLI command implementations."""

from __future__ import annotations

import json
from pathlib import Path

from trainomni.api.evaluate import evaluate
from trainomni.api.export import export_artifact
from trainomni.api.inspect import inspect_task
from trainomni.api.train import train
from trainomni.catalog.local import source_tree_digest


def command_train(args) -> int:
    result = train(
        task_path=args.task,
        run_path=args.run,
        allow_local_code=args.allow_local_code,
        resume_from=args.resume,
        stop_after_steps=args.stop_after_steps,
    )
    print(
        json.dumps(
            {
                "task_digest": result.task_digest,
                "run_digest": result.run_digest,
                "final_step": result.final_step,
                "steps_executed": len(result.records),
            },
            sort_keys=True,
        )
    )
    return 0


def command_inspect(args) -> int:
    print(
        json.dumps(
            inspect_task(args.task, allow_local_code=args.allow_local_code),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_evaluate(args) -> int:
    result = evaluate(
        task_path=args.task,
        run_path=args.run,
        checkpoint=args.checkpoint,
        batches=args.batches,
        allow_local_code=args.allow_local_code,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint),
                "batches": result.batches,
                "samples": result.samples,
                "metrics": result.metrics,
                "receipt": str(result.receipt),
            },
            sort_keys=True,
        )
    )
    return 0


def command_export(args) -> int:
    result = export_artifact(
        task_path=args.task,
        run_path=args.run,
        checkpoint=args.checkpoint,
        exporter=args.exporter,
        destination=args.destination,
        allow_local_code=args.allow_local_code,
    )
    print(
        json.dumps(
            {
                "exporter": result.exporter,
                "checkpoint": str(result.checkpoint),
                "artifact": {
                    "kind": result.artifact.kind,
                    "uri": result.artifact.uri,
                    "digest": result.artifact.digest,
                },
            },
            sort_keys=True,
        )
    )
    return 0


def command_module_digest(args) -> int:
    print(source_tree_digest(Path(args.directory)))
    return 0
