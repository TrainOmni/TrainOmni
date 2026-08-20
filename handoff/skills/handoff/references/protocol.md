# Handoff protocol

Use plain-text envelopes so task messages remain readable and auditable.

## Registration

```text
HANDOFF REGISTER
role: <user-assigned role>
thread_id: <Codex task ID>
scope: <user-assigned responsibility>
```

The coordinator replies:

```text
HANDOFF ACK
role: <role>
thread_id: <task ID>
scope: <responsibility>
```

Registration invariants:

- One role maps to one task.
- One task maps to one role.
- Repeating the same mapping is idempotent.
- Replacement requires an explicit user instruction.
- Discovery never creates a registration.
- Registration contains no inferred fields.
- Delivery uses the registered coordinator task ID. Coordinator title and pin state are not protocol requirements.

After a new registration or user-authorized replacement, the coordinator sends shared context separately from the minimal ACK:

```text
HANDOFF ONBOARDING
to: <newly registered role>
project: <project name>
canonical_root: <current canonical project root>
goal: <shared final goal>
roster: <current roles and scopes>
working_rules: <ownership, routing, and reporting rules>
shared_context: <path or concise context>
```

It then announces the responsibility to registered peers:

```text
HANDOFF NOTICE
id: <stable notice ID>
from: handoff
to: all-registered
kind: roster_update
role: <new or replaced role>
thread_id: <task ID>
scope: <responsibility>
```

Onboarding and roster notices do not add fields to `HANDOFF REGISTER`, and do not turn unregistered recipients into members.

## Routed request

```text
HANDOFF REQUEST
id: <stable request ID>
from: <registered source role>
to: <registered target role>
kind: <question|dependency|blocker|result>
summary: <one-line summary>
context: <facts needed by the target>
requested_action: <what the target should return or do>
```

The target replies through the coordinator:

```text
HANDOFF RESPONSE
id: <same request ID>
from: <registered target role>
to: <registered source role>
status: <answered|accepted|blocked|complete|rejected>
summary: <one-line result>
details: <response or next dependency>
```

## Other messages

```text
HANDOFF STATUS
role: <registered role>
status: <registered|active|idle|needs_input|blocked|complete|stale>
summary: <current state>
```

```text
HANDOFF UNREGISTER
role: <registered role>
reason: <reason>
```

```text
HANDOFF ERROR
code: <stable code>
summary: <human-readable explanation>
```

## Routing rules

- Validate the source and target against the coordinator registry.
- Preserve request IDs across delivery and response.
- Log request, delivery, response, and failure events.
- Send only the context required for the target to act.
- Return missing-target and delivery errors to the source.
- Never enroll, replace, or contact another task merely because it appears relevant.
- Address the coordinator and workers by registered task ID. Never require an exact title or pin state.
- Do not solicit routine phase reports; route real needs and clearly completed, evidenced milestones.
