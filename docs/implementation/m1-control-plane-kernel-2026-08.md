# M1 Control-plane Kernel Implementation

> 历史实现切片：本文记录早期 M1 状态，其中“尚未实现”和“下一切片”不代表当前能力。当前实现、验证和能力边界以 [Framework v1 implementation](framework-v1-2026-08.md)、[support matrix](../design/support-matrix-v1.md) 和 [verification report](../verification/verification-2026-08-20.md) 为准。

- 状态：Implemented slice 1
- 日期：2026-08-20
- 范围：不加载模型权重、不执行训练；先实现配置、插件、能力、artifact、objective/engine 合同和检查 CLI

## 1. 已实现

### Versioned contracts

- `trainomni.contracts`：结构化 issue/report、`ArtifactRef`、`ArtifactManifest`、四种显式 resume level。
- `trainomni.models`：`ModelPluginManifest`、扩展后的 model capabilities、component exact-cover、插件结构/组件 conformance。
- `trainomni.objectives`：`ObjectiveManifest`、requirements、命名 `LossTerm/LossOutput`。
- `trainomni.engines`：`LoopEngine`/`DelegatedStageEngine` 共用 manifest、capability negotiation 和 adapter envelope。

协议版本：

```text
trainomni.model-plugin.v1
trainomni.objective.v1
trainomni.engine.v1
trainomni.artifact.v1
trainomni.run.v1
trainomni.resolved-run.v1
```

### Strict recipe

`trainomni.config` 使用 Pydantic strict public models 和 `yaml.safe_load`：

- public schema 未知字段报错；
- backend/model-specific 参数只能进入显式 `config` namespace；
- component policy、预算、precision、stage type 和 media count 做静态验证；
- user run + 精确 plugin identity 生成稳定 SHA-256 fingerprint；
- recipe requirements 与 model capabilities 在加载权重前协商。

当前配置是单 stage `RunSpec`。Pipeline DAG 按 requirements v1 属于 M2；本轮没有用一个未经验证的 DAG 实现提前污染 StageSpec。

### Plugin registry and trust boundary

`ModelPluginRegistry` 支持：

- Python entry-point candidate 只列出、不自动 import；
- 用户通过 `--plugin MODULE:ATTR` 或 `--plugin FILE.py:ATTR` 显式信任加载；
- plugin ID/version/API/capabilities/methods 校验；
- 同 ID 冲突拒绝；
- 第三方 plugin package 后续使用 `trainomni.model_plugins` entry-point group。

Recipe 本身不能指定要执行的 Python `_target_`，因此读取 YAML 不会自动加载代码。

### CLI

已实现：

```text
trainomni validate CONFIG
trainomni plugins list
trainomni inspect model CONFIG
trainomni dry-run CONFIG
```

也可运行 `python -m trainomni`。所有命令支持全局 `--json`；外部插件必须用全局 `--plugin` 显式提供。

`dry-run` 当前只进行静态解析与 capability validation，输出中明确标记：

```json
{
  "will_load_weights": false,
  "will_execute_training": false
}
```

这避免“dry-run 通过”被误认为 Trainer 已实现。

## 2. Core-zero-edit 证据

测试插件位于 `tests/plugins/toy_vlm_plugin.py`，不在 `src/trainomni` 内，也没有写入 core registry。CLI 通过显式文件路径加载它，然后完成：

- manifest/capabilities 检查；
- component catalog exact-cover；
- strict YAML 解析；
- capability negotiation；
- validate/inspect/dry-run。

示例 recipe：`configs/examples/toy_alignment.yaml`。

## 3. 验证命令

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src'

python -m unittest discover -s tests -v

python -m trainomni `
  --plugin tests/plugins/toy_vlm_plugin.py:PLUGIN `
  validate configs/examples/toy_alignment.yaml

python -m trainomni `
  --plugin tests/plugins/toy_vlm_plugin.py:PLUGIN `
  --json dry-run configs/examples/toy_alignment.yaml
```

2026-08-20 证据：

- 25/25 unit tests passing；
- 所有 Python 文件 `ast.parse` 通过；
- `pyproject.toml` 由 `tomllib` 解析通过；
- 外部插件未显式授权时 CLI 返回非零并拒绝 recipe。
- 相同 recipe 在不同 `PYTHONHASHSEED` 和独立 CLI 进程中产生相同 fingerprint。

实际 CLI 冒烟曾发现 Pydantic 将 `frozenset` 投影为无序 JSON list，导致独立进程 fingerprint 不一致。实现已改为在 hash 前对 typed value 做递归 canonicalization，并加入跨进程回归测试。这个修复说明 fingerprint 必须基于保留类型语义的 resolved object，不能直接 hash 一次普通 `model_dump(mode="json")`。

## 4. 当前不声称完成

- 还没有真实 PyTorch/Transformers 模型加载；
- 还没有 `inspect data/batch`、reader/importer registry；
- 还没有 masked causal-LM objective 实现；
- 还没有 torch single/DDP loop；
- 还没有 pipeline DAG、stateful data、DCP 或 exact-resume 数值等价测试；
- engine/objective 只有稳定合同，尚无 runtime registry/adapter。

## 5. 下一实现切片

1. reader/importer contract、canonical JSONL reader 与 `inspect data`；
2. model plugin encode/collate trace 与 `inspect batch`；
3. objective/engine registry 和内建 masked causal-LM objective；
4. 安装经过固定版本的 torch/transformers，使用公开 tiny VLM 做真实 conformance；
5. 在此基础上实现 TrainOmni-owned single/DDP 2-step loop 和 deterministic save/reload oracle。
