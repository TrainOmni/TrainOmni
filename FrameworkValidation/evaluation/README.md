# VLMEvalKit validation overlay

This directory validates VLMEvalKit as an external TrainOmni evaluation backend.
It is not imported by TrainOmni Core and it does not vendor VLMEvalKit.

## Upstream identity

- Repository: `https://github.com/open-compass/VLMEvalKit.git`
- Commit: `e8e78f05f3080fe28154f2130321f17951c3be94`
- Checkout: `D:\Codex\TrainOmniTemp\framework-upstream-references-20260821\upstreams\VLMEvalKit`
- Python: existing `D:\Codex\TrainOmni\Framework\.venv`; no additional environment

## Windows compatibility overlay

`patches/vlmevalkit-windows.patch` contains two narrowly scoped fixes:

1. make the HiPhO verifier importable when Windows has no `SIGALRM`;
2. copy result aliases when Windows denies unprivileged symbolic-link creation.

`Polygon3` is deliberately omitted because Python 3.12 on this Windows host has
no wheel and building it requires MSVC. Benchmarks requiring it remain unsupported
on this host until that native dependency is supplied. `rouge-score` is installed
explicitly because an eagerly imported upstream module needs it but the upstream
requirements file does not declare it.

Install or verify the pinned checkout:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_vlmevalkit_windows.ps1
```

Run the deterministic image/MCQ smoke:

```powershell
D:\Codex\TrainOmni\Framework\.venv\Scripts\python.exe .\vlmevalkit_smoke.py
```

The smoke uses two real image files and validates model construction, multimodal
prompt construction, two inference calls, prediction persistence, exact-match
evaluation, metric persistence, status reporting, and Windows result publishing.
It fails unless `split=none|Overall` is exactly `1.0` and the upstream status has
no error.

Run the real `stage-05-final` checkpoint on all 247 `AI2D_MINI` samples:

```powershell
D:\Codex\TrainOmni\Framework\.venv\Scripts\python.exe .\vlmevalkit_ai2d_real.py
```

`vlmevalkit-ai2d-real-config.json` pins the real model artifact, both base
checkpoints, CUDA BF16 execution, the benchmark class, and deterministic greedy
decoding. The script fails closed on artifact identity mismatch, incomplete
dataset execution, missing outputs, or model-call count mismatch. The primary
receipt is under `runs-real/TrainOmniStage05AI2D/<eval-id>`.
