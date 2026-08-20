# VLMTraining ownership

- The user-assigned `VLMTrainer` task owns this directory.
- Other tasks treat this directory as read-only unless the user explicitly expands their scope.
- Cross-task dependencies and results are routed through `$handoff`.
- Use `D:\Codex\TrainOmni\VLMTraining` as the canonical path after the root migration.

