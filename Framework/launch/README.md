# Platform launch boundary

This directory is the only shell-specific entry boundary in TrainOmni. The
Framework Python API and CLI do not contain PowerShell, POSIX-shell, path-layout,
environment-activation, scheduler, or host-bootstrap logic.

## Stable contract

Both platform adapters implement the same semantic operation:

```text
<platform-adapter> <trainomni CLI arguments...>
```

They must:

1. require an explicit absolute Python executable in `TRAINOMNI_PYTHON`;
2. invoke that executable as `python -m trainomni <arguments...>`;
3. forward arguments without interpreting TaskSpec or RunSpec;
4. preserve the TrainOmni process exit code;
5. never activate, create, install, or mutate an environment;
6. never select a CUDA/CPU/PyTorch build, device, model, dataset, output path,
   rendezvous endpoint, or distributed backend.

The explicit Python requirement prevents a shell PATH difference from silently
selecting a CPU-only or otherwise incorrect environment.

## Windows

PowerShell is the supported Windows adapter:

```powershell
$env:TRAINOMNI_PYTHON = 'D:\absolute\path\to\python.exe'
& .\launch\windows\trainomni.ps1 inspect --task D:\task\task.yaml
& .\launch\windows\trainomni.ps1 train --task D:\task\task.yaml --run D:\run\run.yaml
```

It uses PowerShell argument-array forwarding and returns `$LASTEXITCODE`. It does
not assume Conda, `venv`, CUDA, or a repository-relative interpreter.

## Linux

The Linux adapter is a POSIX shell script:

```sh
TRAINOMNI_PYTHON=/absolute/path/to/python \
  ./launch/linux/trainomni.sh inspect --task /task/task.yaml
TRAINOMNI_PYTHON=/absolute/path/to/python \
  ./launch/linux/trainomni.sh train --task /task/task.yaml --run /run/run.yaml
```

It uses `exec` so the Python process receives signals directly and becomes the
process supervised by a terminal, container, Slurm step, or other orchestrator.

## Deliberate separation from distributed launch

These two files start one TrainOmni controller/worker process. They do not wrap
`torchrun`, Slurm, `mpirun`, Kubernetes, or Ascend launchers. Distributed process
creation gets its own adapters under `launch/<platform>/distributed/` only after
the Python distributed runtime contract is executable.

The future split is fixed now:

```text
launch/
├── windows/
│   ├── trainomni.ps1
│   └── distributed/             # PowerShell/Windows process adapter
├── linux/
│   ├── trainomni.sh
│   └── distributed/             # torchrun/Slurm/container adapters
└── README.md
```

Distributed topology remains RunSpec semantics. Host-only facts such as node
rank, rendezvous address and scheduler allocation are launch inputs and must be
recorded in a launch receipt, not copied into TaskSpec. A distributed adapter may
translate host facts to upstream launcher arguments, but it may not implement a
training loop or checkpoint policy.

No distributed script is published yet: the current Python distributed modules
are placeholders, so an apparently working multi-process shell wrapper would be
unsafe. Linux validation will add the Linux adapter tests and executable-mode
verification without changing the Python training interface.
