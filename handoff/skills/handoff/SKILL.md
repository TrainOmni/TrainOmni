---
name: handoff
description: Coordinate explicitly assigned Codex tasks in the current project. Use when the user explicitly tells a task to register with handoff, asks handoff to route a question, dependency, blocker, or result between registered tasks, requests registered-task status, or asks to unregister. Never auto-register discovered tasks or infer assignments from titles, pin state, paths, or activity.
---

# Handoff

Coordinate only tasks that the user explicitly enrolled.

## Respect the boundary

- Resolve the current project root from the active project or the registry, and treat `<project-root>/handoff` as coordinator-owned.
- Treat every other project path as read-only unless the user explicitly assigned that path to the current task.
- Never discover, infer, or auto-enroll tasks from titles, activity, or a project path.
- Keep registration minimal: `role`, `thread_id`, and `scope`.
- Route worker work; do not perform it in the coordinator.

Read [references/protocol.md](references/protocol.md) before composing or interpreting a handoff message.

## Determine the current side

Act as the coordinator only when the current task/thread ID equals `coordinator.thread_id` in `<project-root>/handoff/state/registry.json`. Otherwise act as a worker.

The coordinator title and pin state are UI conveniences only. They are neither required for delivery nor proof of coordinator identity. Never block registration or routing because the coordinator is unpinned or has a title other than `handoff`.

## Register from a worker

1. Confirm that the user explicitly assigned the current task a role and told it to register.
2. Obtain the current task ID from Codex task metadata; never guess an ID.
3. Read `coordinator.thread_id` from the project registry when the file is available. Otherwise use a coordinator task ID explicitly provided by the user or in trusted project onboarding context. If no stable ID is available, ask the user; do not search by title or pin state.
4. Send the registration directly to that task ID:

   ```text
   HANDOFF REGISTER
   role: <user-assigned role>
   thread_id: <current task ID>
   scope: <user-assigned responsibility>
   ```

5. Wait for `HANDOFF ACK` or `HANDOFF ERROR`.

Never edit coordinator state from a worker.

## Register at the coordinator

1. Require all three registration fields.
2. Reject blank or inferred fields and registration without an explicit user assignment.
3. Reject role or task conflicts unless the user explicitly ordered replacement.
4. Run:

   ```text
   <python> scripts/registry.py register --role <role> --thread-id <id> --scope <scope>
   ```

   Add `--replace` only for an explicit user-ordered replacement.
5. Send a protocol ACK containing the accepted mapping.
6. When the result is a new registration or an explicitly authorized replacement, separately send the new task `HANDOFF ONBOARDING` with current shared project context, then send every registered task a `HANDOFF NOTICE` describing the new role and scope. Log both deliveries. Do not repeat onboarding or roster broadcast for an idempotent registration unless the user asks.

## Route a request

1. Accept requests only from a registered source to a registered target.
2. Assign or preserve a stable request ID.
3. Log the request before delivery:

   ```text
   <python> scripts/registry.py event --type request --from-role <source> --to-role <target> --request-id <id> --summary <summary>
   ```

4. Read only enough target status to route accurately.
5. Send the request to the registered target with Codex task messaging.
6. Log delivery or failure and report it to the source.
7. If the target is absent, notify the source. Never discover a substitute.

Use native task messaging. Do not move tasks between environments or create worktrees as part of handoff.

## Track registered tasks

Monitor only registered tasks. Use these status values:

- `registered`
- `active`
- `idle`
- `needs_input`
- `blocked`
- `complete`
- `stale`

Silence never counts as registration. Do not message an unregistered task unless the user explicitly instructs it.

Do not request routine intermediate status. Tasks should report only a real dependency, question, blocker, decision requiring user input, a user-requested status, or a clearly completed key result with evidence.

## Maintain state

Only the coordinator writes state. Registry data lives under `<project-root>/handoff/state`. Treat `coordinator.thread_id` as the stable delivery address; `coordinator.title` is display metadata.

Use the bundled workspace Python returned by the Codex workspace-dependency loader. Available commands are:

```text
<python> scripts/registry.py init --project-root <path> --thread-id <id> --title handoff --host-id <host>
<python> scripts/registry.py register --role <role> --thread-id <id> --scope <scope> [--replace]
<python> scripts/registry.py unregister --role <role> --reason <reason>
<python> scripts/registry.py status --role <role> --value <status>
<python> scripts/registry.py event --type <type> --from-role <role> [--to-role <role>] [--request-id <id>] --summary <summary>
<python> scripts/registry.py migrate-root --from-root <old-path> --to-root <new-path>
<python> scripts/registry.py list
<python> scripts/registry.py validate
```
