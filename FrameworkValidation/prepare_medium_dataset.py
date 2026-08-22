"""Materialize deterministic canonical TrainOmni views of the pinned parquet bundle."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

SEED = 20260820
ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(r"D:\Codex\TrainOmni\Downloads\datasets\vlm-minimal-v1\raw")
OUTPUT = ROOT / "data" / "medium-v1"

SOURCES = {
    "diagram": {
        "path": SOURCE_ROOT
        / "diagram_image_to_text"
        / "train-00000-of-00001-37a8de19cc7bc987.parquet",
        "sha256": "d0746661fda57558837ac4b12b4bcd5a3a80915a4d134ad1e1f70c2bb969ea87",
        "rows": 300,
        "train": 240,
        "validation": 30,
    },
    "intergps": {
        "path": SOURCE_ROOT
        / "intergps"
        / "train-00000-of-00001-d12182a583de4589.parquet",
        "sha256": "c7a5c28feec2513fae08c259ccdb8e7d41d3313d20d61b401331705bb2d8c38d",
        "rows": 1280,
        "train": 1024,
        "validation": 128,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": path.relative_to(ROOT).as_posix(), "rows": len(rows), "sha256": sha256(path)}


def image_blocks(dataset: str, row_index: int, images: list[dict]) -> list[dict]:
    blocks = []
    directory = OUTPUT / "media" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    for image_index, payload in enumerate(images):
        raw = payload.get("bytes")
        if not isinstance(raw, bytes) or not raw:
            raise ValueError(f"{dataset} row {row_index} image {image_index} has no bytes")
        path = directory / f"{row_index:05d}-{image_index:02d}.png"
        if not path.exists():
            with Image.open(io.BytesIO(raw)) as image:
                image.convert("RGB").save(path, format="PNG", optimize=True)
        blocks.append({"kind": "image", "value": path.relative_to(ROOT).as_posix()})
    return blocks


def split_indices(row_count: int, train: int, validation: int) -> tuple[list[int], list[int]]:
    indices = list(range(row_count))
    random.Random(SEED).shuffle(indices)
    return indices[:train], indices[train : train + validation]


def conversation(row: dict, dataset: str, row_index: int) -> tuple[list[dict], str, str]:
    texts = row["texts"]
    if not texts:
        raise ValueError(f"{dataset} row {row_index} has no conversation")
    item = texts[0]
    user = str(item.get("user", "")).strip()
    assistant = str(item.get("assistant", "")).strip()
    if not user or not assistant:
        raise ValueError(f"{dataset} row {row_index} has empty text")
    return image_blocks(dataset, row_index, row["images"]), user, assistant


def rejected_answer(chosen: str) -> str:
    match = re.search(r"\bAnswer:\s*([A-D])\b", chosen, flags=re.IGNORECASE)
    if match is None:
        return "This answer is intentionally incorrect."
    options = "ABCD"
    selected = match.group(1).upper()
    return f"Answer: {options[(options.index(selected) + 1) % len(options)]}"


def main() -> None:
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "source_root": str(SOURCE_ROOT),
        "sources": {},
        "outputs": {},
    }
    canonical: dict[str, dict[str, list[dict]]] = {}
    for dataset, spec in SOURCES.items():
        path = spec["path"]
        observed = sha256(path)
        if observed != spec["sha256"]:
            raise ValueError(f"source digest changed for {dataset}: {observed}")
        table = pq.read_table(path)
        if table.num_rows != spec["rows"]:
            raise ValueError(f"source row count changed for {dataset}")
        rows = table.to_pylist()
        train_indices, validation_indices = split_indices(
            table.num_rows, spec["train"], spec["validation"]
        )
        canonical[dataset] = {}
        for split, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
        ):
            converted = []
            for index in indices:
                images, user, assistant = conversation(rows[index], dataset, index)
                if dataset == "diagram":
                    sample = {
                        "sample_id": f"diagram-{index:05d}",
                        "content": [*images, {"kind": "text", "value": assistant}],
                        "metadata": {"prompt": user, "source_row": index},
                    }
                else:
                    sample = {
                        "sample_id": f"intergps-{index:05d}",
                        "messages": [
                            {
                                "role": "user",
                                "content": [*images, {"kind": "text", "value": user}],
                            },
                            {
                                "role": "assistant",
                                "content": [{"kind": "text", "value": assistant}],
                            },
                        ],
                        "metadata": {"source_row": index},
                    }
                converted.append(sample)
            canonical[dataset][split] = converted
            receipt = write_jsonl(OUTPUT / f"{dataset}-{split}.jsonl", converted)
            manifest["outputs"][f"{dataset}-{split}"] = receipt
        manifest["sources"][dataset] = {
            "path": str(path),
            "sha256": observed,
            "rows": table.num_rows,
            "train_indices_sha256": hashlib.sha256(
                json.dumps(train_indices).encode("ascii")
            ).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(
                json.dumps(validation_indices).encode("ascii")
            ).hexdigest(),
        }

    # Cache-heavy objectives use a larger-than-smoke but disk-bounded subset.
    for split, limit in (("train", 64), ("validation", 16)):
        kd = []
        dpo = []
        for source in canonical["intergps"][split][:limit]:
            user_message, assistant_message = source["messages"]
            images = [block for block in user_message["content"] if block["kind"] == "image"]
            prompt = next(
                block["value"]
                for block in user_message["content"]
                if block["kind"] == "text"
            )
            answer = assistant_message["content"][0]["value"]
            suffix = source["sample_id"].split("-", 1)[1]
            kd.append(
                {
                    "sample_id": f"medium-kd-{suffix}",
                    "content": [*images, {"kind": "text", "value": answer}],
                    "metadata": {"prompt": prompt, "source_row": source["metadata"]["source_row"]},
                }
            )
            dpo.append(
                {
                    "sample_id": f"medium-dpo-{suffix}",
                    "messages": [{"role": "user", "content": user_message["content"]}],
                    "metadata": {
                        "chosen": answer,
                        "rejected": rejected_answer(answer),
                        "source_row": source["metadata"]["source_row"],
                    },
                }
            )
        manifest["outputs"][f"kd-{split}"] = write_jsonl(
            OUTPUT / f"kd-{split}.jsonl", kd
        )
        manifest["outputs"][f"dpo-{split}"] = write_jsonl(
            OUTPUT / f"dpo-{split}.jsonl", dpo
        )

    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256(manifest_path)}))


if __name__ == "__main__":
    main()
