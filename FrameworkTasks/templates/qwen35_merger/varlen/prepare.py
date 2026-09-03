"""Create a self-contained task from local paths, refusing to overwrite identities.

No import of another task, no writes to model/data sources, no downloads.
Re-running with identical paths/options/assets is a verification-only operation.
"""

import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from trainomni.catalog.local import source_tree_digest

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def immutable_write(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"Refusing to overwrite {path.name}; copy to a NEW date_task directory first")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def write_json(path, value):
    immutable_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def assets(root):
    root = Path(root)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        names = sorted(set(json.loads(index_path.read_text(encoding="utf8"))["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    if not names or any(Path(name).name != name for name in names):
        raise SystemExit("safetensors index must contain flat relative filenames")
    files = {name: sha(root / name) for name in names}
    # Include tokenizer vocab, chat template, processor and model config, not just weights.
    config_files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix in {
        ".json", ".jinja", ".model", ".txt", ".tiktoken",
    })
    auxiliary = {path.name: sha(path) for path in config_files}
    if "config.json" not in auxiliary:
        raise SystemExit("asset directory must contain config.json")
    return {"weights": files, "auxiliary": auxiliary}


def split_data(source, options):
    rows = next(pq.ParquetFile(source).iter_batches(batch_size=24)).to_pylist()
    if len(rows) != 24:
        raise SystemExit("fixture requires at least 24 InterGPS images/texts rows")
    if options["multi_image_fixture"]:
        # Deliberately exercise 1/2 images per example; this is an engineering fixture.
        for index in (1, 6, 11, 16, 21):
            rows[index]["images"] = rows[index]["images"] + rows[index]["images"]
    files = {}
    for split, values, shards in (("train", rows[:20], 4), ("heldout", rows[20:], 2)):
        for index in range(shards):
            table = pa.Table.from_pylist(values[index::shards])
            buffer = io.BytesIO()
            if options["format"] == "parquet":
                pq.write_table(table, buffer, row_group_size=len(table))
            else:
                with pa.ipc.new_file(buffer, table.schema) as writer:
                    writer.write_table(table)
            payload = buffer.getvalue()
            name = f"{split}-{index:02d}.{options['format']}"
            immutable_write(ROOT / "data" / name, payload)
            files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    manifest = {
        "schema": "trainomni.feedback-fixture.v1", "source_sha256": sha(source),
        "selection": "first 24 rows; train 0:20, heldout 20:24; round-robin physical shards",
        "options": options, "files": files,
    }
    write_json(ROOT / "data" / "manifest.json", manifest)
    return sha(ROOT / "data" / "manifest.json")


def ref(module, config=None):
    return {"module": module, **({} if config is None else {"config": config})}


def pipeline(paths, options, model_assets, split, manifest):
    fmt = options["format"]
    packed = options["packing"]
    return {
        "source": ref(f"data_source:trainomni/{fmt}@1", {
            "dataset_id": f"{ROOT.name}-{split}",
            "paths": [str(ROOT / "data" / f"{split}-*.{fmt}")],
            "columns": ["images", "texts"], "batch_rows": 4,
            "repeat": split == "train", "dataset_manifest_sha256": manifest,
        }),
        "adapter": ref("data_adapter:trainomni/msswift@1", {
            "media_without_placeholders": "prepend", "decode_image_bytes": True,
        }),
        "transforms": [],
        "model_io": ref("model_io:example/qwen35_minicpm5@1", {
            "vision_model_path": paths["vision_model"],
            "language_model_path": paths["language_model"],
            "vision_assets_sha256": model_assets["vision"]["auxiliary"],
            "language_assets_sha256": model_assets["language"]["auxiliary"],
            "min_pixels": 4096, "max_pixels": 16384, "max_tokens": options["max_length"],
        }),
        "supervision": ref("supervision:trainomni/causal_lm@1"),
        "packer": ref("packer:trainomni/sequence@1", {
            "max_length": options["max_length"],
            "max_samples_per_pack": options["max_samples_per_pack"],
            "pad_token_id": 1,
            "concat_fields": ["vision.hidden_states", "vision.grid_thw", "vision.image_counts"],
            "offset_fields": ["modal_positions"],
        }) if packed else ref("packer:trainomni/none@1"),
        "collator": ref("collator:trainomni/multimodal@1", {
            "pad_token_id": 1, "label_pad_id": -100, "padding_side": "right",
            "field_modes": {
                "model_inputs.vision.hidden_states": "concat",
                "model_inputs.vision.grid_thw": "concat",
                "model_inputs.vision.image_counts": "pad", "model_inputs.modal_positions": "pad",
            },
            "field_pad_values": {"model_inputs.vision.image_counts": 0},
        }),
    }


def main():
    paths = json.loads((ROOT / "paths.local.json").read_text(encoding="utf8"))
    if set(paths) != {"vision_model", "language_model", "source_parquet"}:
        raise SystemExit("paths.local.json requires exactly the three paths in paths.example.json")
    paths = {key: str((ROOT / path).resolve()) for key, path in paths.items()}
    options = json.loads((ROOT / "options.json").read_text(encoding="utf8"))
    if options["format"] not in {"parquet", "arrow"} or type(options["packing"]) is not bool:
        raise SystemExit("format must be parquet/arrow; packing must be boolean")
    model_assets = {"vision": assets(paths["vision_model"]), "language": assets(paths["language_model"])}
    write_json(ROOT / "evidence" / "runtime" / "assets.json", model_assets)
    manifest = split_data(paths["source_parquet"], options)
    modules = []
    for module, directory in (
        ("model_io:example/qwen35_minicpm5@1", "model_io"),
        ("encoder:example/qwen35_raw_vit@1", "encoder"),
        ("connector:example/qwen35_merger@1", "connector"),
    ):
        modules.append({"module": module, "path": f"modules/{directory}",
                        "source_sha256": source_tree_digest(ROOT / "modules" / directory)})
    task = {
        "schema_version": 1, "name": ROOT.name, "local_modules": modules,
        "data": pipeline(paths, options, model_assets, "train", manifest),
        "model": {
            "implementation": ref("model:trainomni/composite@1", {
                "branches": [{"name": "vision", "modality": "image", "input_key": "vision",
                              "encoder": "vision", "connector": "connector",
                              "positions_key": "modal_positions", "required": True}],
                "fusion": "fusion", "language": "language",
            }),
            "components": {
                "vision": ref("encoder:example/qwen35_raw_vit@1", {
                    "model_path": paths["vision_model"],
                    "weights_sha256": model_assets["vision"]["weights"],
                    "config_sha256": model_assets["vision"]["auxiliary"]["config.json"],
                }),
                "connector": ref("connector:example/qwen35_merger@1"),
                "fusion": ref("fusion:trainomni/token_replace@1"),
                "language": ref("language:trainomni/transformers_causal_lm@1", {
                    "model_name_or_path": paths["language_model"], "local_files_only": True,
                    "asset_manifest_sha256": digest(model_assets["language"]),
                }),
            },
            **({"attention_policy": ref("attention_policy:trainomni/packed_block_diagonal@1")} if options["packing"] else {}),
        },
        "objective": ref("objective:trainomni/causal_lm@1"),
        "parameters": ref("parameter_policy:trainomni/component@1", {
            "train": ["connector"], "group_per_component": True,
        }),
        "evaluation": {
            "data": pipeline(paths, options, model_assets, "heldout", manifest),
            "evaluators": [ref("evaluator:trainomni/loss@1", {
                "term": "token_ce", "metric_name": "heldout_token_ce",
            })],
        },
        "exporters": [ref("exporter:trainomni/safetensors@1", {"filename": "model.safetensors"})],
    }
    modules.append({"module": "model:example/qwen35_merger_varlen@1", "path": "modules/varlen",
                    "source_sha256": source_tree_digest(ROOT / "modules" / "varlen")})
    task["model"]["implementation"]["module"] = "model:example/qwen35_merger_varlen@1"
    task["model"].pop("attention_policy")
    for data in (task["data"], task["evaluation"]["data"]):
        data["packer"]["module"] = "packer:trainomni/padding_free@1"
        data["collator"]["module"] = "collator:trainomni/padding_free@1"
        for key in ("model_inputs.modal_positions", "model_inputs.vision.image_counts"):
            data["collator"]["config"]["field_modes"][key] = "stack"
    immutable_write(ROOT / "task.yaml", yaml.safe_dump(task, sort_keys=False).encode())
    write_json(ROOT / "evidence" / "runtime" / "prepared.json", {
        "schema": "trainomni.prepared-task.v2", "task_name": task["name"],
        "task_yaml_sha256": sha(ROOT / "task.yaml"), "modules": modules,
        "assets_digest": digest(model_assets), "dataset_manifest_sha256": manifest,
    })
    print(json.dumps({"task": task["name"], "prepared": True, "writes_outside_task": False}))


if __name__ == "__main__":
    main()
