"""Portable two-process CPU DDP smoke launcher (including Windows CI)."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    plugin: str,
    config: str,
    output_dir: str,
    resume: str | None,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["TRAINOMNI_DIST_INIT_METHOD"] = init_method
    from trainomni.cli import main

    argv = [
            "--plugin",
            plugin,
            "--json",
            "train",
            config,
            "--output-dir",
            output_dir,
        ]
    if resume:
        argv.extend(
            ["--resume", resume.format(rank=rank), "--trusted-resume"]
        )
    code = main(argv)
    if code:
        raise RuntimeError(f"rank {rank} exited with code {code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--resume")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = (args.output_dir.resolve() / f".ddp-store-{uuid.uuid4().hex}").as_uri()
    import torch.multiprocessing as mp

    mp.spawn(
        _worker,
        args=(
            args.world_size,
            store,
            args.plugin,
            str(args.config.resolve()),
            str(args.output_dir.resolve()),
            args.resume,
        ),
        nprocs=args.world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
