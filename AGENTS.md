# TrainOmni task coordination

- The coordinator task identified by `coordinator.thread_id` in `handoff/state/registry.json` coordinates explicitly registered project tasks. Its title and pin state are optional UI metadata.
- Register only after the user explicitly assigns this task a role and tells it to register. Invoke `$handoff` and send the minimal registration payload.
- Never auto-register or infer assignments from a project path, a task title, or observed activity.
- Outside the responsibility explicitly assigned by the user, treat project content as read-only.
- Do not modify `handoff` unless this is the coordinator task.
- Route cross-task questions, dependencies, blockers, and results through the coordinator.
- After a new or explicitly replaced registration, handoff provides shared onboarding information to that task and broadcasts its role and scope to registered peers.
- Avoid nonessential phase reports. Report a real dependency, blocker, question, decision requiring input, user-requested status, or a clearly completed key result with evidence.
