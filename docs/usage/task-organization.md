# Task 隔离、命名与历史保留

一个 Task 是一组有明确目的的训练或验证实验，不是整个项目唯一的一份可变
`task.yaml`，也不是一个 Codex 对话。**每组测试建立独立 Task；旧 Task、配置、
结果和证据保留，新实验不得覆盖旧实验。**

## Task 与 Run 的边界

- **Framework**：可复用代码，不保存具体实验的模型路径或生成结果。
- **Task**：这一组实验做什么，包括模型结构、数据与监督语义、objective、
  参数策略和所需的 task-local 模块。
- **Run**：同一个 Task 如何执行一次，包括 seed、设备、精度、batch size、
  worker 数、优化器和步数。同组可以有多个独立 Run。
- **Output**：某次执行实际生成的配置快照、日志、checkpoint 和导出文件。
  每次独立执行使用自己的输出目录；继续原执行才使用 `--resume`。

例如，同一 packing 验证 Task 可以用不同 Run 对照 `num_workers=0/2` 或
`batch_size=1/2`。更换模型结构、objective、参数策略或已验证的数据处理语义，
则另建 Task；不要在已有结果的 Task 上改写原配置来表示新实验。

尚未运行的草稿可以编辑。一旦某份 Task/Run 配置产生需要保留的结果，就保留
这份配置和对应模块版本；新配置不得继续使用旧结果作为自己的验证证据。

## 强烈建议：日期_具体任务

目录名和 `TaskSpec.name` 强烈建议一致，使用：

```text
YYYYMMDD_<具体任务>
```

日期是该实验建立的日期，推荐后半部分使用小写英文、数字和下划线，写清
模型变体及实验目的。例如：

```text
20260904_qwen35_merger_alignment
20260904_qwen35_merger_sft
20260904_qwen35_merger_lora_sft
20260904_qwen35_merger_packing
20260904_qwen35_merger_kd
20260904_qwen35_merger_dpo
```

这是强烈建议的组织约定，不是新的 schema 限制。同日同目的但独立的新实验
可以增加 `_v2` 或更具体的差异后缀。已有不符合命名约定的 Task 不需要为了
统一名称而删除、移动或改名；它们及其原始证据继续保留。

Task 可以按项目、模型或测试类别采用层级目录组织，例如
`tasks/qwen35_merger/packing/20260904_batch2_workers2/`；上层目录仅用于归类，
实际 Task 目录仍强烈建议采用 `YYYYMMDD_<具体任务>` 命名，并保持配置和结果独立。

## 推荐目录

以下是目录组织示例，不表示这些真实模型任务已建立或验证完成。实际实验目录
放在 Framework 外的调用方工作区；任务定义、Run 配置和输出各有独立子目录，
可以随整组实验一起归档：

```text
<experiment-root>/tasks/
├── <existing-task>/                    # 原测试：原结构、配置和证据保留
├── 20260904_qwen35_merger_alignment/
│   ├── README.md                       # 目的、路径配置、命令、结果与限制
│   ├── task.yaml                       # 这一组实验的 TaskSpec
│   ├── modules/                        # 本 Task 的 hash-pinned 适配代码
│   ├── assets/                         # 资产身份清单，不复制整套权重
│   ├── data/                           # 小 fixture 或数据清单/划分
│   ├── runs/
│   │   ├── smoke.yaml                  # RunSpec：首次最小运行
│   │   └── workers2.yaml               # RunSpec：同 Task 的另一运行配置
│   ├── outputs/                        # 生成物，通常不提交 Git
│   │   ├── smoke_001/
│   │   └── workers2_001/
│   └── evidence/                       # 可提交的小型验收记录
├── 20260904_qwen35_merger_sft/
└── 20260904_qwen35_merger_packing/
```

独立 Task 不意味着复制虚拟环境、全部模型或大数据集。可以引用同一份只读、
有版本/摘要的资产；不要依赖另一个 Task 的临时脚本、可变配置或未声明的输出。
若下一阶段需要上一阶段的 checkpoint，必须显式记录该 artifact 的身份与来源。

Run 与输出目录也可以放在 Task 目录之外；CLI 不要求上述布局。无论采用哪种
物理布局，都不能让独立实验共享一个可写输出根。

## 路径与执行

`--task` 和 `--run` 都显式指定。task-local 模块路径相对于 `task.yaml` 所在
目录；相对的 `checkpoint.directory` 则相对于 **Run 配置文件所在目录**，
不是当前 shell 目录。采用上面的布局时，`runs/smoke.yaml` 中写：

```yaml
schema_version: 1
name: smoke
# 其余 RunSpec 字段按该 Task 的完整运行配置填写。
checkpoint:
  directory: ../outputs/smoke_001/checkpoints
  every_steps: 2
```

以下命令在该 Task 根目录执行，前提是完整配置与可信本地模块已经准备好：

```text
trainomni inspect --task task.yaml --allow-local-code
trainomni train --task task.yaml --run runs/smoke.yaml --allow-local-code
```

独立重跑时为新 Run 配置选择新的输出目录，例如
`../outputs/smoke_002/checkpoints`，无需删除 `smoke_001`。
`RunSpec.name` 只是一项配置，**不会自动生成或隔离输出目录**。
恢复原运行使用原 Task/Run 的语义配置与明确的 checkpoint；不能把改变模型
结构后的 Task 当作旧 checkpoint 的 exact resume。

遇到 `resolved run identity already differs`，先核对是否误用了其他实验的输出
根，不要删除旧结果或强行覆盖 identity 文件。新实验使用新输出根，原实验的
恢复按 [quickstart](quickstart.md) 的 checkpoint 合同执行。

## 新模型与可复用模板的保留规则

Qwen ViT + 原 merger + 附加线性投影的旧测试 Task 和历史证据保留。
Qwen 预训练 ViT + 重新初始化的 merger connector + LLM 是**新 Task**：旧的
附加投影不进入新模型，但不因此删除旧 Task、旧代码快照或旧 artifact。

模板来自实际跑通的某个 Task，而不是另写一份未验证的近似配置。交付前应：

1. 保留原 Task 的验证记录；复制并脱敏得到独立模板，不覆盖原 Task。
2. 集中说明需要替换的模型、数据、输出路径，不包含本机私有绝对路径。
3. 提供完整 task-local 模块、配置、准备及运行命令，不引用相邻 Task 的代码。
4. 将模板复制到另一个 `YYYYMMDD_<具体任务>` 目录，替换路径和 Task 名称，
   按说明重新验证；记录这一份副本自己的配置身份与结果。
5. README 写明 Framework 版本、依赖、输入资产、验证范围、证据位置和已知
   限制。历史通过、新模型通过、未验证的服务器行为不得混为一谈。

模板需要的资产校验或生成步骤必须有明确命令。单纯调整路径或目录组织不会
自动完成复现，也不会把旧模型的验证结果转移给新模型。
