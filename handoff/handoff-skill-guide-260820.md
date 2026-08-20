# Codex 多 Session Handoff Skill：可复用设计与实施指南

> 一份项目无关的经验总结，用于在同一项目内协调多个由用户明确分工的 Codex task/session。本文既描述设计原则，也给出可直接复用的目录、协议、Skill 模板和实施检查表。

## 目录

1. [要解决的问题](#1-要解决的问题)
2. [核心设计](#2-核心设计)
3. [职责与权限边界](#3-职责与权限边界)
4. [推荐目录结构](#4-推荐目录结构)
5. [注册模型](#5-注册模型)
6. [消息协议](#6-消息协议)
7. [状态与请求生命周期](#7-状态与请求生命周期)
8. [状态文件与注册表工具](#8-状态文件与注册表工具)
9. [在新项目中实现](#9-在新项目中实现)
10. [可复用的 Skill 模板](#10-可复用的-skill-模板)
11. [AGENTS.md 模板](#11-agentsmd-模板)
12. [日常使用流程](#12-日常使用流程)
13. [跨设备协作](#13-跨设备协作)
14. [验证与测试](#14-验证与测试)
15. [常见错误](#15-常见错误)
16. [扩展方向](#16-扩展方向)
17. [最终检查表](#17-最终检查表)

## 1. 要解决的问题

Codex 的多个 task/session 可以并行负责一个项目的不同部分，但它们默认没有稳定的团队名册、责任边界和跨 task 请求状态。仅靠用户手工复制消息，容易出现以下问题：

- 不知道某个角色当前对应哪个 task。
- 某个 task 有依赖或疑问，却不知道该交给谁。
- 收到消息被误认为任务已经完成。
- 单个请求完成后，整个角色被错误标记为完成。
- task 根据目录、标题或活动情况自行“发现同事”，造成越权和误注册。
- 多个 task 同时修改共享状态，导致注册表冲突。
- 跨设备同步时出现多个状态源，无法判断哪个状态可信。

Handoff Skill 的目标不是替代项目管理系统，也不是让协调器承包其他 task 的工作。它提供的是一层很薄的协调框架：

1. 用户明确分配角色和范围。
2. worker 主动向 handoff 注册。
3. handoff 维护唯一名册并路由结构化消息。
4. 请求和回复使用稳定 ID 串联。
5. 所有重要变化写入可审计事件日志。

它适合一个用户同时运行若干 Codex task、各 task 有明确目录或职责边界、需要偶尔互相提问或交付结果的场景。

## 2. 核心设计

### 2.1 显式注册，不做自动发现

注册的前提必须同时满足：

- 用户已经明确指定该 task 的角色和 scope；
- 用户要求该 task 向 handoff 注册；
- 注册消息由该 task 发给 handoff。

不能根据 task 标题、工作目录、最近活动或猜测自动登记。handoff 也不能主动扫描其他 task 并招募它们。

这一规则比便利性更重要：task 的存在不等于用户授权它加入当前协作关系。

### 2.2 单一写入者

只有 handoff 协调 task 可以写注册表和事件日志。worker 只能发送协议消息，不能直接修改状态文件。

这样可获得三个好处：

- 写入顺序清晰；
- 注册冲突可以集中检查；
- 事件日志可以作为统一审计线索。

### 2.3 角色与 task 一一对应

在同一个项目中维持以下不变量：

- 一个 role 最多对应一个 thread/task ID；
- 一个 thread/task ID 最多对应一个 role；
- 相同映射重复注册是幂等操作；
- 冲突映射默认拒绝；
- 替换只能由用户明确授权。

角色名应简短、稳定、描述责任，而不是描述一次性动作。例如 `data`、`framework`、`evaluation`、`release`，而不是 `fix-bug-123`。

### 2.4 路由元数据，不搬运大文件

handoff 消息适合传递：

- 问题、依赖和 blocker；
- 文件路径、revision、hash 和验证状态；
- 简短结论和下一步动作；
- 请求是否已接受或完成。

模型、数据集、构建产物等大文件仍保留在其责任 task 管理的位置。消息传元数据和定位信息，不传载荷本身。

### 2.5 协调器不是项目负责人

handoff 只负责：

- 注册与注销；
- 查询名册与状态；
- 路由请求、回复和状态更新；
- 记录事件；
- 报告无法投递、冲突和超时迹象。

handoff 不应：

- 代替 worker 实现业务功能；
- 未经用户授权重新分配 scope；
- 修改 worker 负责目录；
- 根据自己判断注册新角色；
- 把“联系过某个 task”视为“该 task 已注册”。

## 3. 职责与权限边界

推荐把协作关系定义为三层：

| 主体 | 可写范围 | 主要职责 |
|---|---|---|
| 用户 | 决定所有角色和 scope | 分工、授权替换、处理重大冲突 |
| handoff task | handoff 自有目录和原生 task 消息 | 名册、路由、日志、状态汇总 |
| worker task | 用户明确指定的目录或职责范围 | 执行业务工作、汇报状态、响应请求 |

对 worker 的默认规则是：

- 自己负责范围可写；
- 其他项目范围优先只读；
- 需要跨范围修改时，先向对应角色发请求，或请用户重新授权；
- 公共目录是否可写应由项目另行定义，不属于注册协议的一部分。

未来新增的汇总、发布或 GitHub 交接角色，也只是普通 worker。用户分配后再注册，不需要在 handoff 中硬编码特殊身份。

## 4. 推荐目录结构

一种适合“协调器只拥有 handoff 目录”的布局是：

```text
<PROJECT_ROOT>/
├─ AGENTS.md
├─ .agents/
│  └─ skills/
│     └─ handoff -> ../../handoff/skills/handoff
└─ handoff/
   ├─ AGENTS.md
   ├─ state/
   │  ├─ registry.json
   │  └─ events.jsonl
   └─ skills/
      └─ handoff/
         ├─ SKILL.md
         ├─ agents/
         │  └─ openai.yaml
         ├─ references/
         │  └─ protocol.md
         └─ scripts/
            └─ registry.py
```

这里把实际 Skill 文件放在 handoff 自有目录，再通过项目级 `.agents/skills/handoff` 暴露给所有 task。这样既符合 Codex 的项目 Skill 发现约定，又不会让多个 worker 共同拥有 Skill 源文件。

如果项目不需要严格的目录所有权，也可以直接把 Skill 放在：

```text
<PROJECT_ROOT>/.agents/skills/handoff/
```

两种布局只能选一种作为真实来源，避免维护两份副本。

### 4.1 创建发现链接

macOS/Linux 可使用符号链接：

```bash
mkdir -p <PROJECT_ROOT>/.agents/skills
ln -s ../../handoff/skills/handoff <PROJECT_ROOT>/.agents/skills/handoff
```

Windows PowerShell 可使用目录联接：

```powershell
New-Item -ItemType Directory -Force -Path '<PROJECT_ROOT>\.agents\skills'
New-Item -ItemType Junction `
  -Path '<PROJECT_ROOT>\.agents\skills\handoff' `
  -Target '<PROJECT_ROOT>\handoff\skills\handoff'
```

如果目标已经存在，先确认它究竟是正确链接、普通目录还是用户文件；不要直接覆盖。

## 5. 注册模型

### 5.1 用户给 worker 的指令

用户可以使用类似指令：

```text
你负责 <role>，scope 是 <明确的目录或职责>。
现在使用 $handoff 向 handoff task 注册。
```

### 5.2 最小注册消息

用户提供的语义字段只需要三个：

```text
HANDOFF REGISTER
role: <role>
thread_id: <task/thread id>
scope: <明确职责或负责路径>
```

不要要求用户或 worker 手工填写时间、状态、host ID 等内部字段。它们由协调器自动生成。

### 5.3 注册处理顺序

handoff 收到注册后应按顺序执行：

1. 校验 `role`、`thread_id`、`scope` 非空。
2. 检查 role 与 thread ID 的一一对应约束。
3. 相同映射则幂等确认，并在需要时更新 scope。
4. 冲突映射则拒绝，不自动替换。
5. 先持久化注册表和事件，再回复确认。
6. 回复中明确“已注册”，不要暗示业务工作已经完成。

### 5.4 替换与注销

替换角色对应 task 是高影响操作，只能在用户明确要求后执行。推荐先记录注销原因，再登记新 task：

```text
HANDOFF UNREGISTER
role: <role>
reason: <用户授权的替换或结束原因>
```

同一 role 的新旧 task 不应在注册表里同时处于有效状态。

## 6. 消息协议

协议应保持文本化、短小、可复制，并能被人和程序读取。正文可以使用自然语言，但头部字段必须稳定。

### 6.1 请求

```text
HANDOFF REQUEST
id: <stable-request-id>
from: <registered-role>
to: <registered-role>
kind: question|dependency|blocker|result
summary: <一行摘要>
context: <必要背景、现有证据、相关路径>
requested_action: <希望对方具体做什么>
```

规则：

- `id` 在整个往返过程中保持不变；
- `from` 和 `to` 默认都必须已注册；
- `requested_action` 应可执行，不要只写“看看”；
- 依赖要说明交付物、格式、位置和完成判定；
- blocker 要说明阻塞对象和绕过方案是否存在。

### 6.2 回复

```text
HANDOFF RESPONSE
id: <same-request-id>
from: <registered-role>
to: <registered-role>
status: answered|accepted|blocked|complete|rejected
summary: <一行结论>
details: <答案、交付位置、验证证据或下一步>
```

状态含义：

| 状态 | 含义 | 请求是否关闭 |
|---|---|---|
| `answered` | 已回答问题；若无后续动作即可结束 | 通常是 |
| `accepted` | 已接受，工作仍在进行 | 否 |
| `blocked` | 已接收但当前无法继续 | 否 |
| `complete` | 请求中的交付动作已完成 | 是 |
| `rejected` | 请求无效、越界或无法接受，并给出原因 | 是 |

### 6.3 角色状态

```text
HANDOFF STATUS
role: <registered-role>
status: registered|active|idle|needs_input|blocked|complete|stale
summary: <当前工作、进度、依赖或下一步>
```

角色状态是对整个 worker 工作状态的概括，不是某一个请求的状态。

### 6.4 ACK

```text
HANDOFF ACK
id: <request-or-message-id>
from: <role>
to: <role>
summary: <已收到、已记录或已接受投递>
```

`ACK` 只表示消息已收到、已记录或已接受处理，绝不表示请求完成。完成必须使用 `HANDOFF RESPONSE`，并给出 `status: complete` 或适用的最终状态。

### 6.5 错误

```text
HANDOFF ERROR
id: <request-id-if-any>
code: <stable-error-code>
summary: <错误摘要>
details: <冲突、缺失角色或投递失败原因>
```

推荐错误码包括：

- `ROLE_NOT_REGISTERED`
- `THREAD_NOT_REGISTERED`
- `ROLE_CONFLICT`
- `THREAD_CONFLICT`
- `INVALID_MESSAGE`
- `DELIVERY_FAILED`
- `REPLACEMENT_NOT_AUTHORIZED`

## 7. 状态与请求生命周期

### 7.1 两套生命周期必须分离

Handoff 中最容易出错的是把请求状态和角色状态混为一谈。

```text
角色生命周期：registered -> active <-> idle/needs_input/blocked -> complete/stale

请求生命周期：created -> delivered -> accepted/blocked -> answered/complete/rejected
```

示例：某个 worker 完成了“提供 checkpoint 清单”这个请求，可以回复 `status: complete`；但它仍在负责后续下载工作，所以角色状态仍应是 `active`，不能被自动改成 `complete`。

### 7.2 推荐请求状态机

即使第一版只使用事件日志，也应在语义上遵守以下状态：

```text
created
  ├─> delivery_failed
  └─> delivered
        ├─> accepted ─> complete
        ├─> blocked  ─> accepted/complete/rejected
        ├─> answered
        └─> rejected
```

`accepted` 不是终态。handoff 需要保留该请求为 open，直到收到终态回复或用户取消。

### 7.3 投递顺序

推荐顺序是：

1. 验证 source 和 target 注册状态。
2. 写入 `request_created` 事件。
3. 调用 Codex 原生 task 消息能力异步投递。
4. 成功后写入 `request_delivered`；失败则写入 `delivery_failed`。
5. 目标回复时，以同一个 request ID 写入响应事件。
6. 根据回复状态更新 open-request 索引，但不隐式改变角色状态。

如果目标未注册，返回明确错误，不要搜索“可能是它”的其他 task。

### 7.4 用户直接授权的例外

用户可以明确要求 handoff 临时联系一个未注册 task，例如邀请它做团队介绍。此时可以投递，但必须满足：

- 这次联系有直接用户授权；
- 联系行为不产生注册；
- 未注册 task 仍不能作为普通路由目标；
- 后续注册仍需走完整 `HANDOFF REGISTER`。

## 8. 状态文件与注册表工具

### 8.1 `registry.json`

推荐最小结构：

```json
{
  "schema_version": 1,
  "project_root": "<PROJECT_ROOT>",
  "coordinator": {
    "thread_id": "<COORDINATOR_THREAD_ID>",
    "title": "handoff",
    "host_id": "<HOST_ID>",
    "created_at": "<ISO-8601>"
  },
  "members": {
    "<role>": {
      "thread_id": "<WORKER_THREAD_ID>",
      "scope": "<SCOPE>",
      "status": "registered",
      "registered_at": "<ISO-8601>",
      "updated_at": "<ISO-8601>"
    }
  }
}
```

其中 `status`、时间戳和 coordinator 元数据是内部字段，不增加注册消息的负担。

### 8.2 `events.jsonl`

事件日志采用 append-only JSON Lines，每行一个事件：

```json
{"at":"<ISO-8601>","type":"request_delivered","from_role":"<role-a>","to_role":"<role-b>","request_id":"<id>","summary":"<summary>"}
```

建议记录：

- coordinator 初始化；
- 注册、幂等注册、注销和替换；
- 角色状态变更；
- 请求创建、投递和投递失败；
- ACK、响应和请求关闭；
- 校验失败和注册冲突。

不要在事件中写入密钥、访问令牌或不必要的敏感内容。

### 8.3 `registry.py` 命令契约

注册表脚本建议至少提供：

```text
init       --project-root --thread-id --title handoff --host-id
register   --role --thread-id --scope [--replace]
unregister --role --reason
status     --role --value
event      --type --from-role [--to-role] [--request-id] --summary
list
validate
```

还应支持 `--state-dir`，以便在临时目录中进行隔离测试。

实现必须满足：

- JSON 使用临时文件、flush/fsync 和原子替换写入；
- JSONL 追加后 flush/fsync；
- 所有输入经过 schema 和一一对应校验；
- 冲突时返回非零退出码；
- 不通过 shell 字符串拼接执行用户输入；
- `list` 和 `validate` 是只读操作；
- 错误信息可被人理解，也有稳定错误类型。

小项目可以只用注册表和事件日志。并发请求增多后，建议增加 `requests.json` 或从事件日志构建 open-request 索引，显式记录 `created/delivered/accepted/blocked/complete/failed`。

## 9. 在新项目中实现

### 第一步：确定唯一协调 task

创建或指定一个 Codex 协调 task，并明确它只负责 handoff 目录。将其 task/thread ID 记录为唯一稳定地址；标题可以使用 `handoff` 或更易识别的项目化名称，是否置顶也不影响协议。标题与置顶状态只能辅助人类查找，不能用于身份校验或阻止投递。

### 第二步：生成 Skill 骨架

优先使用 Codex 自带的 `skill-creator` 初始化脚本创建 `handoff` Skill，并选择 `scripts,references` 资源目录。不要手工创建一堆与 Skill 无关的辅助 README。

逻辑上等价于：

```text
init_skill.py handoff \
  --path <PROJECT_ROOT>/handoff/skills \
  --resources scripts,references \
  --interface display_name=Handoff \
  --interface short_description="Coordinate registered Codex tasks" \
  --interface default_prompt="Use $handoff to register this explicitly assigned task or route a cross-task request."
```

具体脚本路径以当前 Codex 的 `skill-creator` 安装位置为准，不要把某台机器的用户目录写进项目。

### 第三步：写入 Skill 与协议

- `SKILL.md`：只写触发条件、强约束和工作流程；
- `references/protocol.md`：写完整消息格式和状态语义；
- `scripts/registry.py`：实现确定性的注册表操作；
- `agents/openai.yaml`：提供简洁的 UI 名称、描述和默认 prompt。

### 第四步：建立项目级发现入口

将 Skill 直接放入 `.agents/skills/handoff`，或从该位置链接到 handoff 自有目录。完成后重新启动或刷新 Codex，使新 Skill 被 task 发现。

### 第五步：初始化状态

```text
<PYTHON> registry.py --state-dir <PROJECT_ROOT>/handoff/state init \
  --project-root <PROJECT_ROOT> \
  --thread-id <COORDINATOR_THREAD_ID> \
  --title handoff \
  --host-id <HOST_ID>
```

PowerShell 中可写成一行，或使用反引号续行；macOS/Linux shell 使用反斜杠续行。

### 第六步：配置 AGENTS.md

项目根 `AGENTS.md` 声明全局协作规则；handoff 目录内 `AGENTS.md` 声明协调器的更严格边界。模板见下一节。

### 第七步：做隔离验证

不要先用真实注册表试错。使用临时 `--state-dir` 验证初始化、注册、幂等、冲突、状态更新、事件追加和 schema 校验。

### 第八步：由用户开始分配和注册

Skill 安装完成不等于任何 worker 已注册。等待用户明确分配角色，再由对应 task 发起注册。

## 10. 可复用的 Skill 模板

下面的 `SKILL.md` 可作为项目无关基线：

```markdown
---
name: handoff
description: Coordinate explicitly assigned Codex tasks in one project. Use when the user explicitly tells a task to register with handoff, asks handoff to route a question, dependency, blocker, or result between registered tasks, requests registered-task status, or asks to unregister. Never auto-register discovered tasks or infer assignments from titles, pin state, paths, or activity.
---

# Handoff

Coordinate only tasks that the user explicitly assigned and instructed to register.

## Hard rules

- Never discover or auto-register tasks.
- Accept registration only from an explicitly assigned task.
- Keep one role mapped to one task, and one task mapped to one role.
- Treat identical registration as idempotent; reject conflicts unless the user explicitly authorizes replacement.
- Only the handoff coordinator writes registry and event state.
- Do not modify worker-owned project areas or perform their business work.
- Route ordinary requests only when both source and target roles are registered.
- A user-authorized one-off contact with an unregistered task does not register it.
- ACK means received or recorded, not completed.
- Keep request state separate from overall role state.

## Registration

Require exactly these semantic fields:

```text
HANDOFF REGISTER
role: <role>
thread_id: <thread id>
scope: <scope>
```

Generate status and timestamps internally. Persist a valid registration before acknowledging it.

## Routing

1. Validate source and target against the registry.
2. Record the request event before delivery.
3. Deliver through the native Codex task messaging capability.
4. Record delivery success or failure.
5. Preserve the request ID through all ACK and RESPONSE messages.
6. Keep accepted or blocked requests open until a terminal response arrives.
7. Never mark a role complete merely because one request completed.

Read `references/protocol.md` for message formats and state semantics. Use `scripts/registry.py` for deterministic registry mutations and validation.
```

Skill 的 description 是触发入口，应同时说明“什么时候用”和“什么时候绝不能自动用”。正文则负责约束执行细节。

### 10.1 `agents/openai.yaml` 模板

```yaml
interface:
  display_name: "Handoff"
  short_description: "Coordinate registered Codex tasks and route requests"
  default_prompt: "Use $handoff to register this explicitly assigned task or route a cross-task request."
```

## 11. AGENTS.md 模板

### 11.1 项目根规则

```markdown
# Task coordination

- The task whose ID equals `coordinator.thread_id` in the registry coordinates explicitly registered project tasks; its title and pin state are optional UI metadata.
- Do not infer task assignments from task titles, directories, or activity.
- A worker registers only after the user explicitly assigns its role and scope and asks it to register.
- Each worker may write only within its explicitly assigned scope; other project areas are read-only unless the user expands that scope.
- Workers do not edit handoff registry or event files directly.
- Cross-task questions, dependencies, blockers, results, and status updates use `$handoff` and the structured protocol.
- ACK means received, not completed. A completed request does not automatically complete the worker role.
```

### 11.2 handoff 目录规则

```markdown
# Handoff coordinator

- This task owns only the handoff directory and coordination state.
- It registers only tasks explicitly assigned by the user and sent through the registration protocol.
- It is the sole writer of registry and event state.
- It routes messages between registered roles and records delivery results.
- It does not perform worker business tasks or modify worker-owned directories.
- It rejects conflicting registration unless the user explicitly authorizes replacement.
- It does not auto-contact or auto-register unregistered tasks.
- Request lifecycle and role lifecycle are tracked separately.
```

如果项目已有 `AGENTS.md`，合并这些规则，不要覆盖用户已有指令。

## 12. 日常使用流程

### 12.1 注册

```text
Worker -> Handoff: HANDOFF REGISTER
Handoff: 校验映射并持久化
Handoff -> Worker: HANDOFF ACK（已注册）
Handoff -> Worker: HANDOFF ONBOARDING（共享目标、根路径、名册和工作规则）
Handoff -> 已注册同事: HANDOFF NOTICE（新角色及 scope）
```

只有首次注册或用户明确授权的替换需要 onboarding 与全员责任播报；相同映射的幂等注册不重复打扰同事。注册消息本身仍只包含 role、thread ID 和 scope。

### 12.2 发起依赖

```text
HANDOFF REQUEST
id: api-schema-001
from: implementation
to: research
kind: dependency
summary: 请求确认输入 schema
context: 当前实现需要冻结 v1 字段；草案位于 <path>。
requested_action: 请返回必填字段、可选字段和一份有效示例。
```

handoff 验证后投递，并将 `api-schema-001` 记为 open。

### 12.3 接受但尚未完成

```text
HANDOFF RESPONSE
id: api-schema-001
from: research
to: implementation
status: accepted
summary: 已接受，正在核对边界条件
details: 预计在完成官方接口核验后返回最终字段表。
```

此时请求仍未关闭。

### 12.4 完成交付

```text
HANDOFF RESPONSE
id: api-schema-001
from: research
to: implementation
status: complete
summary: v1 schema 已确认
details: 字段表和示例位于 <path>，校验命令为 <command>，测试结果为 <result>。
```

handoff 关闭该请求，但不会自动把 `research` 角色标记为 `complete`。

### 12.5 状态查询

用户询问“现在谁在做什么”时，handoff 应返回注册表中的角色、scope、最新状态和 open 请求摘要。不要把未注册 task 加入 roster，也不要把陈旧事件包装成当前事实；必要时明确“最后更新时间”。

## 13. 跨设备协作

### 13.1 推荐模式：单一执行主机

最简单可靠的方式是让有完整项目文件和计算资源的主机作为唯一执行主机：

- handoff 注册表和 worker task 都以该主机为准；
- 另一台设备通过 Codex 的远程连接能力访问同一主机；
- 不复制 task ID，不维护第二份运行时注册表；
- Git 只同步静态 Skill、协议、脚本和项目代码。

这种模式保留单一事实源，特别适合一台工作设备负责交互、另一台高性能设备负责实验的情况。

### 13.2 真正的多主机模式

如果 task 确实分布在多个独立主机，需要额外设计 transport endpoint：

- 内部记录 `host_id` 或可路由 endpoint；
- 仍保持一个权威注册表写入者；
- 投递层根据 host 路由，而不是让各主机独立修改同一 JSON；
- 注册消息的用户语义字段仍保持 `role/thread_id/scope`，host 信息由系统补充；
- 网络分区时记录投递失败，不猜测远端状态。

不要用普通 Git 合并两个主机同时写入的 `registry.json` 和 `events.jsonl`。这会破坏顺序和唯一性。

### 13.3 Git 中提交什么

建议提交：

- `SKILL.md`
- `references/protocol.md`
- `scripts/registry.py`
- `agents/openai.yaml`
- 相关 `AGENTS.md`

通常不提交运行时状态：

- `state/registry.json`
- `state/events.jsonl`
- 含 thread ID、主机路径和工作摘要的快照

可在版本库中保留不含真实数据的 `.example` 文件，或让初始化命令首次生成状态。

### 13.4 是否需要 Git worktree

多个 session 在同一工作区、并且各自只写明确目录时，一般不需要 worktree。worktree 更适合：

- 多个 task 必须同时修改同一批 Git 跟踪文件；
- 每个 task 需要独立分支、独立 index 或独立构建树；
- 某项工作要完整迁移到另一台主机。

它不是 handoff 注册和消息路由的前提。

## 14. 验证与测试

### 14.1 Skill 静态验证

使用 `skill-creator` 提供的校验脚本检查 Skill：

```text
quick_validate.py <path-to-handoff-skill>
```

至少检查：

- frontmatter 只有允许字段；
- `name` 与目录名一致；
- description 包含明确触发条件；
- 引用的脚本和 reference 实际存在；
- `agents/openai.yaml` 中 `$handoff` 没有被 shell 展开或错误转义。

### 14.2 注册表隔离测试

在临时目录依次验证：

1. `init` 生成有效空注册表。
2. 首次注册成功。
3. 相同 role/thread 重复注册不产生冲突。
4. 同 role 换 thread 默认失败。
5. 同 thread 换 role 默认失败。
6. 未经授权不能 `--replace`。
7. 合法状态更新成功，非法状态被拒绝。
8. 事件为合法 JSONL，顺序与操作一致。
9. `validate` 能发现手工破坏的 schema。
10. 并发或中断写入不会留下半个 JSON 文件。

### 14.3 协议行为测试

至少模拟：

- 向未注册目标发请求，得到 `ROLE_NOT_REGISTERED`；
- `accepted` 后请求仍为 open；
- `complete` 后只关闭对应请求；
- ACK 不关闭请求；
- 用户直接联系未注册 task 后，roster 保持不变；
- 投递失败被记录，且不会伪造对方已收到；
- 注销后普通路由拒绝该角色。

### 14.4 前向测试

让一个没有参与实现的 Codex task 仅阅读本指南，尝试在临时项目中搭建 Skill。观察它是否会：

- 错误自动发现 task；
- 把 ACK 当完成；
- 把请求完成当角色完成；
- 要求用户提供额外注册字段；
- 直接编辑 JSON；
- 在多设备上创建多个权威状态源。

如果发生这些行为，应优先修改指令中导致误解的部分，而不是只增加更多例子。

## 15. 常见错误

### 15.1 自动注册“看起来相关”的 task

问题：越过用户授权，也可能把旧 task 或无关 task 加入项目。

修正：只有明确的 `HANDOFF REGISTER` 才改变 roster。

### 15.2 在注册消息里堆叠项目规则

问题：注册协议越来越重，不同 worker 填写不一致。

修正：注册只保留 role、thread ID、scope。公共目录、代码风格和发布规则放在项目 `AGENTS.md` 或对应文档中。

### 15.3 worker 直接写注册表

问题：并发冲突、无法审计、可能破坏一一对应关系。

修正：worker 发消息，handoff 单独写状态。

### 15.4 ACK 被当成完成

问题：请求被过早关闭，依赖方继续工作后才发现交付物不存在。

修正：ACK 只确认接收；最终交付必须是 RESPONSE 终态。

### 15.5 请求完成后把角色标记为完成

问题：一个长期角色通常承担多个请求。

修正：请求状态与角色状态使用不同字段和不同更新入口。

### 15.6 未注册依赖被自动联系

问题：记录“需要某角色”不等于用户已创建或授权该角色。

修正：只记录依赖，并告知用户当前 target 未注册。除非用户直接要求，否则不联系、不创建、不注册。

### 15.7 状态文件通过 Git 双向同步

问题：事件顺序、thread ID 和主机信息会冲突，还可能泄露内部元数据。

修正：运行时状态只有一个权威副本；Git 同步静态实现。

### 15.8 handoff 开始替 worker 干活

问题：职责边界被打破，协调状态与业务修改混在一起。

修正：handoff 只路由。如果缺少 worker，等待用户分配，或向用户报告阻塞。

## 16. 扩展方向

第一版保持简单即可。项目规模扩大后，可以按需要增加：

### 16.1 Open request 索引

从事件日志重建或维护 `requests.json`，支持：

- 按 source/target/状态查询；
- 检测长时间停留在 `accepted` 或 `blocked` 的请求；
- 生成交接清单；
- 防止同一终态重复关闭。

### 16.2 心跳与 stale 检测

由 worker 主动发送 `HANDOFF STATUS`。handoff 可以根据最后更新时间提示 `stale`，但不能仅凭沉默断言 task 已失败或自动注销。

### 16.3 汇总与发布角色

项目后期可由用户新增一个专门负责成果汇总、版本整理和 GitHub 交接的角色。handoff 只把各角色的最终结果、路径、验证证据和 open blocker 路由给它，不自行承担发布工作。

### 16.4 外部项目管理系统

当请求数量超出文本事件日志的可管理范围，可把 handoff 映射到 issue tracker 或数据库。但仍应保留：

- 用户显式注册；
- 单一角色映射；
- 稳定 request ID；
- ACK 与完成分离；
- 请求和角色生命周期分离；
- 唯一权威状态源。

外部系统是持久化或展示层，不应改变授权语义。

## 17. 最终检查表

部署前确认：

- [ ] 注册表保存唯一 coordinator thread ID；标题和置顶状态不参与身份校验。
- [ ] handoff 只拥有自己的目录。
- [ ] 项目级 Skill 能被所有 task 发现。
- [ ] `SKILL.md` 明确禁止自动发现和自动注册。
- [ ] 注册语义字段只有 role、thread ID、scope。
- [ ] role 与 task 是一一对应关系。
- [ ] 只有 handoff 能写注册表和事件日志。
- [ ] 请求使用稳定 ID。
- [ ] ACK 明确定义为“收到”，不是“完成”。
- [ ] 请求生命周期和角色生命周期分离。
- [ ] 未注册 target 默认不可路由。
- [ ] 用户直接联系未注册 task 不会产生隐式注册。
- [ ] 状态写入具有原子性，事件日志可审计。
- [ ] 已用临时状态目录完成冲突和失败测试。
- [ ] 多设备只有一个权威运行时状态源。
- [ ] Git 只同步适合公开的静态实现，不同步敏感运行时状态。
- [ ] 新增角色仍由用户明确分配后注册。

## 参考

- [OpenAI：Build skills](https://learn.chatgpt.com/docs/build-skills)
- [OpenAI：AGENTS.md configuration](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [OpenAI：Remote connections](https://learn.chatgpt.com/docs/remote-connections)

这套方案的本质是：用户掌握组织权，worker 掌握业务范围，handoff 掌握协调状态。三者分离后，多 task 协作才能既灵活，又可追踪、可恢复、可跨项目复用。
