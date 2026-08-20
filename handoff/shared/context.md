# TrainOmni shared task context

## Project identity and goal

- Project: `TrainOmni`
- Canonical Windows root: `D:\Codex\TrainOmni`
- Coordinator task ID: `01a01a50-ef47-7a12-bd3d-ec9a123d89e0`
- Coordinator title and pin state are optional UI metadata. Direct task ID is the stable handoff address.
- Shared goal: integrate target models, sufficient research, executable training routes, framework implementation, and experimental evidence into a reproducible, easy-to-use, verifiable, and sustainably transferable omni-modal training project.
- Delivery order: VLM first; audio understanding next; diffusion and generation later unless the user changes priority.
- A future user-assigned role will consolidate outcomes and handle GitHub/release handoff. It is not registered until the user explicitly assigns and registers it.

## Registered roster

- `framework` (`01a01a87-b774-7102-b7bc-777244261c49`): owns `D:\Codex\TrainOmni\Framework`; reusable framework research, architecture, implementation, verification, and documentation.
- `VLMTrainer` (`01a01d8a-779a-7692-a08d-353c3ed7907a`): owns `D:\Codex\TrainOmni\VLMTraining`; builds and runs real VLM training tasks using framework capabilities and research strategy.
- `downloader` (`01a01a47-df36-7843-9eca-34c514e48ebb`): owns `D:\Codex\TrainOmni\Downloads`, `D:\Models`, and later download directories explicitly assigned by the user; downloads, records, validates, and archives assets.

The `research` task currently remains unregistered. Its existence or project activity does not create a handoff role.

## Working rules

- Write only within the scope explicitly assigned by the user. Other project areas are read-only unless authorization is expanded.
- Route cross-task questions, dependencies, blockers, and results through `$handoff` using registered task IDs.
- Do not require the coordinator to be titled exactly `handoff` or pinned.
- Do not send routine phase updates. Report a real dependency, blocker, question, decision needing input, a status explicitly requested by the user, or a clearly completed key result with evidence.
- `HANDOFF ACK` means received or recorded; it does not mean the underlying request is complete.
- New registration remains minimal (`role`, `thread_id`, `scope`). Handoff sends this shared context separately and broadcasts the new responsibility to registered peers.

