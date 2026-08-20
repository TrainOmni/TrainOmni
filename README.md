# TrainOmni

TrainOmni is an omni-modal training project with a VLM-first delivery path. Audio understanding follows after the VLM path is stable; diffusion and generation are deferred until explicitly scheduled.

Current areas:

- `Framework/`: reusable TrainOmni training framework.
- `VLMTraining/`: real VLM training integration and experiments built on the framework.
- `Research/`: model and training-strategy research.
- `Downloads/`: reproducible asset manifests; large model files live outside this repository.
- `handoff/`: cross-task coordination skill, protocol, and shared context.

The canonical and only project root is `D:\Codex\TrainOmni`.
