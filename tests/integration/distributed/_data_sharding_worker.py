from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import torch

from trainomni.core.context import BuildContext
from trainomni.core.module import ModuleRef
from trainomni.modules.data.sources.parquet.module import descriptor


def _source(path: Path, rank: int, world_size: int):
    reference = ModuleRef.from_mapping(
        {
            "module": "data_source:trainomni/parquet@1",
            "config": {
                "dataset_id": "two-rank-fixture",
                "paths": [str(path)],
                "repeat": True,
            },
        },
        field_name="data.source",
    )
    source = descriptor().build(reference, BuildContext("task"))
    source.shard(rank=rank, world_size=world_size)
    return source


def _rank_main(
    rank: int,
    world_size: int,
    phase: str,
    root_text: str,
    data_text: str,
    port: int,
) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "USE_LIBUV": "0",
        }
    )
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    root = Path(root_text)
    data = Path(data_text)
    try:
        source = _source(data, rank, world_size)
        if phase == "capture":
            prefix = [source.next_record().sample_id for _ in range(3)]
            state = source.state_dict()
            expected = [source.next_record().sample_id for _ in range(6)]
            local = {"rank": rank, "prefix": prefix, "state": state, "expected": expected}
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, local)
            if rank == 0:
                (root / "capture.json").write_text(
                    json.dumps(gathered, sort_keys=True),
                    encoding="utf-8",
                )
        elif phase == "resume":
            captured = json.loads((root / "capture.json").read_text(encoding="utf-8"))
            saved = captured[rank]
            source.load_state_dict(saved["state"])
            actual = [source.next_record().sample_id for _ in range(6)]
            local = {
                "rank": rank,
                "expected": saved["expected"],
                "actual": actual,
                "state": source.state_dict(),
            }
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, local)
            if rank == 0:
                (root / "result.json").write_text(
                    json.dumps(gathered, sort_keys=True),
                    encoding="utf-8",
                )
        else:  # pragma: no cover - test controls the phase
            raise ValueError(f"unknown phase: {phase}")
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def main() -> None:
    phase = sys.argv[1]
    root = Path(sys.argv[2]).resolve()
    data = Path(sys.argv[3]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    torch.multiprocessing.spawn(
        _rank_main,
        args=(2, phase, str(root), str(data), port),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()
