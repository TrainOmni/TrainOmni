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

## Distributed launch

The single-process files above never wrap a process launcher. PyTorch DDP,
FSDP2, and the thin DeepSpeed adapter use separate torchrun wrappers:

```powershell
& .\launch\windows\distributed\torchrun.ps1 -NProcPerNode 1 `
  train --task D:\task.json --run D:\run.json
```

```sh
./launch/linux/distributed/torchrun.sh --nproc-per-node 8 -- \
  train --task /task.json --run /run.json
```

Multi-node host facts are launcher inputs (node count, rendezvous address/port,
and Linux static node rank). The selected execution backend and expected world size remain strict
RunSpec fields. The wrappers never inspect a task or run file.

The future split is fixed now:

```text
launch/
├── windows/
│   ├── trainomni.ps1
│   └── distributed/torchrun.ps1
├── linux/
│   ├── trainomni.sh
│   └── distributed/torchrun.sh
└── README.md
```

Distributed topology remains RunSpec semantics. Host-only facts such as node
rank, rendezvous address and scheduler allocation are launch inputs and must be
recorded in a launch receipt, not copied into TaskSpec. A distributed adapter may
translate host facts to upstream launcher arguments, but it may not implement a
training loop or checkpoint policy.

Native Windows has Gloo but no NCCL/libuv in the validated PyTorch build, so its
wrapper directly starts one worker and is certified only for world-size-one backend probes. Real CUDA
multi-device execution belongs on Linux/NCCL. The Linux wrapper is source-checked
on Windows; its executable multi-process gate remains a Linux-server test.
