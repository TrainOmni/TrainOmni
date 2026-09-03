# 真实 VLM feedback 复现入口

本轮模型是 **Qwen3.5 预训练 raw ViT → 全新 Qwen patch merger → MiniCPM5-1B**。
ViT 的 147 个 state tensors 从 checkpoint 加载；原 merger 不加载、不执行。
新 connector 直接复用上游 `Qwen3_5VisionPatchMerger`：
`LN(768) → 2x2 grouping → Linear(3072,3072) → GELU → Linear(3072,1536)`。
Linear 权重按固定 seed 随机初始化，bias 为零，LayerNorm 为默认初始化。
不再叠加旧单层 MLP。通用 Framework 中的 linear connector 和旧 task 不删除。

## 从真实模板开始

模板位于项目根的 `FrameworkTasks/templates/qwen35_merger/`：

| 子目录 | 用途 | 本地验收范围 |
| --- | --- | --- |
| `unpacked` | 最小端到端起点 | BF16、batch2、worker0/1/2、训练/保存/评估/导出/重载 |
| `packed` | Parquet dense packing | batch2 packs、worker0/1/2、FP32 数值 oracle、BF16 样本隔离 |
| `arrow` | Arrow IPC dense packing | 同上，独立 Arrow reader/worker 路径 |
| `varlen` | 显式 padding-free | 1 pack 含多样本、upstream CUTLASS、无 dense LM mask |

每份都有完整 `modules/`、`prepare.py`、`options.json` 和 `runs/`，不 import
相邻任务，也不引用 `FrameworkValidation`/`VLMTraining` 的代码或输出。
复制所需模板到自己的 **新** `YYYYMMDD_具体任务` 目录；可按模型/类别分层。
Task 名称由该目录名生成。同一 Task 的 Run 分别写入独立的 `outputs/` 子目录。

## 只配置实际需要的路径

环境已经可用时不需要再建环境，也不需要安装 ms-swift/TRL。
使用现有 CUDA Python，能 import 本次 Framework 源码、TorchData、Transformers、
safetensors、PyArrow、Pillow、PyYAML；varlen 额外需要兼容的 xFormers。
当前验证版本和 source digest 见[本轮验收报告](../verification/feedback-v3-20260904.md)。

PowerShell 示例（用自己的路径替换三个变量；不复制虚拟环境）：

```powershell
$frameworkRoot = 'X:/your/TrainOmni/Framework'
$pythonExe = 'X:/your/existing/cuda/python.exe'
$taskRoot = 'X:/your/tasks/qwen35_merger/packing/20260904_my_validation'
$env:PYTHONPATH = "$frameworkRoot/src"
Set-Location $taskRoot
Copy-Item paths.example.json paths.local.json
```

在 `paths.local.json` 中填写三项真实路径：

- `vision_model`：完整 Qwen/Qwen3.5-0.8B 目录，含权重、config、image processor。
- `language_model`：完整 openbmb/MiniCPM5-1B 目录，含权重、tokenizer、chat template。
- `source_parquet`：ms-swift 可读 `images`/`texts` 的 InterGPS Parquet 文件。
  本 fixture 要求至少 24 行；不是任意 Parquet 自动具有相同列语义。

可用绝对路径，也可用相对 **Task 根**的路径。输出路径已经是 Task 内相对路径，
无需填入本机私有目录。公开模板不包含 `paths.local.json`、生成的 `task.yaml`、
数据、模型、日志、checkpoint。`prepare.py` 校验完整资产文件摘要并生成这些本地绑定。
它只写当前 Task；重复执行必须内容一致，否则拒绝覆盖，提示另建 Task。

## 完整运行

在 unpacked/packed/arrow 模板的独立副本中执行：

```powershell
& $pythonExe -B run_checks.py
if ($LASTEXITCODE -ne 0) { throw 'validation failed; inspect evidence/runtime/checks' }
```

这不是另一个训练 engine：脚本逐条调用下面的公开 CLI，保存 stdout/stderr 和
返回码。任一步失败立即停止，不删除旧输出、不自动降级。也可以手工运行：

```powershell
& $pythonExe -B prepare.py
& $pythonExe -B -m trainomni inspect --task task.yaml --allow-local-code
& $pythonExe -B -m trainomni train --task task.yaml --run runs/baseline.yaml --allow-local-code
& $pythonExe -B -m trainomni train --task task.yaml --run runs/worker1.yaml --allow-local-code
& $pythonExe -B -m trainomni train --task task.yaml --run runs/worker2.yaml --allow-local-code
& $pythonExe -B verify_model.py
```

packed/arrow 另运行 `verify_model.py --precision fp32`，比较各样本独立 forward。
unpacked baseline 保存了 step2，可接着运行：

```powershell
$checkpoint = 'outputs/baseline_001/checkpoints/step-00000002'
& $pythonExe -B -m trainomni evaluate --task task.yaml --run runs/baseline.yaml --checkpoint $checkpoint --batches 2 --allow-local-code
& $pythonExe -B -m trainomni export --task task.yaml --run runs/baseline.yaml --checkpoint $checkpoint --destination outputs/export_001 --allow-local-code
& $pythonExe -B verify_export.py
```

`run_checks.py` 已执行这些命令，不要再重复导出到同一目标。packed/arrow 的默认
Run 明确关闭 checkpoint，用于有限显存/磁盘上的 data/attention 工程验收，
不可把它们当作可恢复训练。需要 checkpoint 时另建 Run，显式打开，并使用新输出。

varlen 模板使用 `runs/worker0.yaml`、`runs/worker2.yaml` 和 `verify_varlen.py`：

```powershell
& $pythonExe -B prepare.py
& $pythonExe -B -m trainomni train --task task.yaml --run runs/worker0.yaml --allow-local-code
& $pythonExe -B -m trainomni train --task task.yaml --run runs/worker2.yaml --allow-local-code
& $pythonExe -B verify_varlen.py
```

varlen 的 Run 使用 `attention_kernel: auto`，task-local adapter 显式调用 CUTLASS；
它不是 FlashAttention，也不是普通 composite 的通用开关。这里只验证单 GPU。

## 读结果与修改任务

- `outputs/<run>/metrics/events.jsonl`：每步 CE、grad norm、实际参数变化、GPU peak、
  worker wait、时间戳与 metric scope；不要把累计 wait 当作每步或 GPU idle。
- `evidence/runtime/model-oracle-*`：真实 mixed-image batch、raw ViT 权重核对、
  随机 merger、packed 样本边界和隔离证据。
- `evidence/runtime/checks/passed.json`：全部命令返回码；没有该文件不能宣称整组通过。
- `evidence/runtime/export-reload.json`：独立进程严格加载导出，372 tensors 与
  checkpoint 逐项相等，再运行真实 held-out forward。

这 24 行只作工程 fixture：train 前 20 行、held-out 后 4 行，部分样本复制同一
真实图像以验证一/两图混合边界。不能据此宣称模型质量、泛化或吞吐提升。
当前默认只训练新 merger；**不是 full-SFT/KD/DPO 全路线的重新验收**。
需要改 objective/参数策略/数据语义时，复制到新 Task 再准备、运行和验证。

保存过结果的 Task/Run 和失败现场都保留。数据/模型/模块语义变化时不要直接
覆盖 `task.yaml` 或旧输出；读[命名与历史保留规则](task-organization.md)。
本轮不宣称已复现或消除了服务器 Linux/HAMI/NCCL 的原始 SIGSEGV。
