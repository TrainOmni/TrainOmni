# Quickstart

## 1. Install

```powershell
cd D:\Codex\TrainOmni\Framework
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[torch,peft]"
```

`data`、`eval`、`peft` 分组都是 optional；核心 schema/inspect 不强制安装 torch。

## 2. Inspect a recipe without weights

```powershell
$plugin = "examples/plugins/tiny_llava.py:PLUGIN"

trainomni --plugin $plugin validate configs/examples/tiny_llava_smoke.yaml
trainomni --plugin $plugin inspect model configs/examples/tiny_llava_smoke.yaml
trainomni --plugin $plugin inspect data configs/examples/tiny_llava_smoke.yaml --samples 2
trainomni --plugin $plugin inspect batch configs/examples/tiny_llava_smoke.yaml --samples 1
trainomni --plugin $plugin dry-run configs/examples/tiny_llava_smoke.yaml
```

`inspect batch` 允许加载 tokenizer/processor，但不调用 `plugin.build()`，不会加载模型权重。

## 3. Train

```powershell
trainomni --plugin $plugin train configs/examples/tiny_llava_smoke.yaml `
  --output-dir runs/tiny-llava
```

输出包含：

```text
runs/tiny-llava/
  provenance.json
  metrics.jsonl
  run-manifest.json
  checkpoints/
    step-00000001/
    step-00000002/
```

## 4. Resume exactly

Local/DDP exact checkpoints保存 Python pickle runtime state，只对可信训练目录使用：

```powershell
trainomni --plugin $plugin train configs/examples/tiny_llava_smoke.yaml `
  --output-dir runs/tiny-llava-resume `
  --resume runs/tiny-llava/checkpoints/step-00000001 `
  --trusted-resume
```

配置 fingerprint 必须一致。若只需要迁移权重，应走 export/model-only checkpoint，而不是伪装成 exact resume。

## 5. Evaluate and export

```powershell
trainomni --plugin $plugin evaluate configs/examples/tiny_llava_smoke.yaml `
  --checkpoint runs/tiny-llava/checkpoints/step-00000002 `
  --trusted-checkpoint --output-dir runs/tiny-llava-eval --max-batches 2

trainomni --plugin $plugin export configs/examples/tiny_llava_smoke.yaml `
  --checkpoint runs/tiny-llava/checkpoints/step-00000002 `
  --trusted-checkpoint --output-dir runs/tiny-llava-hf --format hf
```

内置本地 evaluator 遵循 recipe 的 `stage.engine.config.device` 和 `stage.engine.precision`：主模型、辅助模型和 batch 会移动到同一 device，并在 `eval()`、`inference_mode()` 与对应 autocast/TF32 上下文中执行。实际解析出的 backend/device/precision 会写入 `evaluation.json` 的 `execution` 字段。

FSDP2 DCP 的 `model_only` 不读取 rank runtime pickle，可直接在单进程重分片并导出，不需要 `--trusted-checkpoint`。

## 6. Pipeline

```powershell
trainomni --plugin $plugin plan configs/examples/tiny_llava_pipeline.yaml
trainomni --plugin $plugin run configs/examples/tiny_llava_pipeline.yaml `
  --output-dir runs/tiny-llava-pipeline
```

边会把前一 stage 的 artifact ID、selector 和物理 URI传到下一 stage。Pipeline state 原子写入 `pipeline-state.json`；恢复已有 exact artifact 时要求 `--trusted-resume`。

## 7. Distributed smoke

生产环境用 `torchrun` 提供 `RANK/LOCAL_RANK/WORLD_SIZE`。Windows CPU CI 可用无 libuv 依赖的 file rendezvous：

```powershell
python scripts/run_local_ddp_smoke.py `
  --plugin tests/plugins/torch_toy_vlm_plugin.py:PLUGIN `
  --config configs/examples/torch_toy_ddp_smoke.yaml `
  --output-dir runs/ddp-smoke
```

FSDP2 只需把 recipe 的 `engine.parallelism` 改为 `fsdp2`；checkpoint 自动切换到 DCP。

## 8. Global flags

- `--plugin FILE.py:ATTR`：显式信任 model plugin，可重复。
- `--data-plugin FILE.py:ATTR`：显式信任 reader/importer plugin，可重复。
- `--json`：machine-readable output。

这些 flag 必须放在 subcommand 前面。
