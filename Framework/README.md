# TrainOmni Framework

TrainOmni 是一个 **VLM-first**、模型中立、数据协议独立、后端可替换的全模态训练框架。当前版本只把视觉语言理解训练做深做稳；音频理解是下一模态扩展，diffusion/图像音视频生成明确延后。核心管理 canonical 多模态语义、模型/数据插件、Stage/Pipeline DAG、能力协商、精确恢复、产物血缘和 CLI；数值执行复用 PyTorch/Transformers/FSDP2/DCP，规模化路径优先适配 VeOmni，复杂 post-training/RL 委托给 TRL、veRL 或 NeMo 等开源后端。

当前逐路状态、upstream foundation、真实 VLM 证据和最小接入合同，以
[`docs/verification/training-path-coverage-2026-08-21.md`](docs/verification/training-path-coverage-2026-08-21.md)
为唯一权威覆盖矩阵。标准能力遵循 upstream-first；核心不再扩写已有开源
trainer、distributed、PEFT、compile 或 quantized optimizer。

当前实现不是目标模型脚本。新 VLM 的正常接入只新增一个外部 Model Plugin，不修改 core、数据系统、训练循环或 CLI。

## 已实现能力

- 完整阶段模型：vision preparation、alignment、multimodal pretraining、curriculum、SFT、distillation、reward/verifier、preference、online/agentic RL、evaluation/export。
- strict YAML/JSON `RunSpec` 与 `PipelineSpec`，未知字段拒绝、稳定 fingerprint、静态 capability negotiation。
- canonical image/video/audio/text/tool/grounding/preference/rollout sample；reader 与 importer 分离。
- JSON/JSONL、Parquet、TAR-JSON reader；显式 `--data-plugin` 可加入任意 reader/importer。
- weighted deterministic mixture、media-aware cost budget、batch planning、跨 rank 确定性 batch sharding、精确 reader/mixture/look-ahead state。
- 外部 Model Plugin：build、capabilities、component exact-cover、encode、collate、export；加载代码必须显式 `--plugin` 授权。
- built-in masked causal LM、native offline full-vocab cached-logit KD 与 native offline-reference sigmoid DPO；live/大规模 distillation/DPO、GRPO、PPO 保留 delegated objective contract。
- PyTorch single/DDP/FSDP2 loop：component freeze/LR/weight decay/dtype/grad clip、gradient accumulation、AMP/TF32、scheduler、per-component activation-checkpoint receipt、LoRA/QLoRA、`torch.compile`。
- 显式 optimizer contract：native AdamW（含 `foreach=false`）、受控 optional AdamW8bit、optimizer class/version/state dtype/quantization metadata，以及不允许 silent fallback 的失败语义。
- P1 训练证据：optimizer 内 trainable numel、逐组件 finite grad norm、全参数 bitwise update scan、CUDA current/peak allocated/reserved，并可配置硬门禁。
- single/DDP 原子 local checkpoint；FSDP2 使用 DCP + rank-local exact runtime state；model-only DCP 可重分片并在单进程导出。
- exact resume 保存 model、optimizer、scheduler、scaler、Python/Torch RNG、step/token、reader/mixture/batch state。
- Pipeline DAG、stage inputs、artifact URI/lineage、metric/artifact/manual gates、持久化状态与恢复。
- internal normalized-loss evaluator、显式授权的 external evaluator、model-plugin-owned export。
- shell-free delegated stage engine；VeOmni 使用要求固定 revision 和 bridge API 的专用 VLM command adapter，TRL、NeMo、veRL/custom 使用通用 adapter。
- JSONL metrics、resolved provenance、run/eval/export/delegated manifests。

详见 [支持矩阵](docs/design/support-matrix-v1.md)、[架构蓝图](docs/design/framework-blueprint-v1.md) 和 [实现报告](docs/implementation/framework-v1-2026-08.md)。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[torch,peft]"
```

仅做 schema、数据、插件和规划检查时安装基础包即可：

```powershell
python -m pip install -e .
```

## 先检查，再训练

```powershell
$plugin = "examples/plugins/tiny_llava.py:PLUGIN"

trainomni --plugin $plugin validate configs/examples/tiny_llava_smoke.yaml
trainomni --plugin $plugin inspect data configs/examples/tiny_llava_smoke.yaml --samples 2
trainomni --plugin $plugin inspect batch configs/examples/tiny_llava_smoke.yaml --samples 1
trainomni --plugin $plugin train configs/examples/tiny_llava_smoke.yaml `
  --output-dir runs/tiny-llava
```

精确恢复、评测与导出：

```powershell
trainomni --plugin $plugin train configs/examples/tiny_llava_smoke.yaml `
  --output-dir runs/tiny-llava-resumed `
  --resume runs/tiny-llava/checkpoints/step-00000001 `
  --trusted-resume

trainomni --plugin $plugin evaluate configs/examples/tiny_llava_smoke.yaml `
  --checkpoint runs/tiny-llava/checkpoints/step-00000002 `
  --trusted-checkpoint --output-dir runs/eval --max-batches 2

trainomni --plugin $plugin export configs/examples/tiny_llava_smoke.yaml `
  --checkpoint runs/tiny-llava/checkpoints/step-00000002 `
  --trusted-checkpoint --output-dir runs/export --format hf
```

Pipeline：

```powershell
trainomni --plugin $plugin plan configs/examples/tiny_llava_pipeline.yaml
trainomni --plugin $plugin run configs/examples/tiny_llava_pipeline.yaml `
  --output-dir runs/tiny-llava-pipeline
```

更多命令见 [Quickstart](docs/usage/quickstart.md)。

## 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

当前证据：80 项自动测试通过（含跨进程 DataSpec exact-resume canonicalization、gradient accumulation exact resume、真实 PEFT LoRA、QLoRA k-bit adapter contract、`torch.compile`、Parquet/TAR 状态恢复、凭据脱敏、两阶段物理 checkpoint 传递、VeOmni bridge、evaluate device/precision、full-parameter SFT P1、offline dense-logit KD，以及 offline-reference DPO identity/numerics/pair/resume contract）；公开 1.05M 参数 tiny LLaVA 完成真实 encode→forward/backward→checkpoint→exact resume→eval→HF export；两进程 CPU DDP 和 FSDP2+DCP 均完成训练及 exact resume 验证。完整命令和结果见 [验证记录](docs/verification/verification-2026-08-20.md)，P1 接入见 [Full-parameter SFT P1](docs/usage/full-parameter-sft-p1.md)，KD 接入见 [Offline dense-logit KD](docs/usage/offline-dense-logit-kd.md)，DPO 接入见 [Offline-reference DPO](docs/usage/offline-reference-dpo.md)。

## 目录

- `src/trainomni/`：框架实现。
- `configs/examples/`：single、DDP、FSDP2、Pipeline 示例。
- `examples/plugins/`：不修改 core 的公开模型插件。
- `tests/plugins/`：dependency-free 与真实 PyTorch conformance 插件。
- `docs/research/`：SOTA 框架调研与 Build-vs-Adopt 决策。
- `docs/design/`：生命周期、需求、协议、支持矩阵。
- `docs/usage/`：使用和扩展指南。
- `docs/verification/`：可交接验证证据。

当前优先级：VLM 训练与目标模型接入 > 音频理解协议和 encoder 接入 > 生成训练。Ascend/昇腾适配按用户要求暂不进入本版本；后续优先通过 VeOmni engine adapter 接入，不在 core 内重写 HCCL/分片训练。
