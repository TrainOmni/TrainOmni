# Windows CUDA development environment

The active local Framework interpreter is:

~~~text
D:\Codex\TrainOmni\Framework\.venv\Scripts\python.exe
~~~

`.venv/` is ignored by Git. It is a local execution environment, not source and
not part of an archive or wheel. Platform launch scripts do not activate it or
guess its location; point `TRAINOMNI_PYTHON` to it explicitly.

## Installed and observed

| Item | Value |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB |
| Compute capability | 8.9 |
| BF16 support | yes |
| Transformers | 5.15.0 |
| safetensors | 0.8.0 |
| TrainOmni | 0.1.0 editable install |

The venv was created from the existing local `cuda13` Python and uses its
already-installed non-Torch packages. CUDA Torch is installed inside `.venv`, so
the selected Torch build is `2.13.0+cu130`, not a CPU wheel. Set
`PYTHONNOUSERSITE=1` for validation to exclude unrelated per-user packages.

## Verification performed

- `torch.cuda.is_available()` returned true.
- `DeviceContext("cuda:0", "bf16_true")` placed a two-layer module and input on
  CUDA in BF16.
- Forward, backward and `torch.optim.AdamW(..., foreach=False).step()` completed.
- Peak allocated/reserved CUDA memory was observable.
- Ruff passed.
- Tiny composite BF16 train/checkpoint/fresh-process exact resume/evaluate/export
  passed on CUDA with deterministic weighted multi-source input.
- Tiny monolithic BF16 train/checkpoint/resume/evaluate/export passed on CUDA.
- Full source suite: 88 passed, 1 platform skip.
- `pip check`: no broken requirements with user site packages disabled.
- `python -m trainomni --help`: passed.

These checks establish a working Windows CUDA development environment and complete
tiny-model framework lifecycles. They do not claim that a real VLM checkpoint fits,
trains correctly, or reaches any throughput target; those require a task-level
integration run.

## Launch example

~~~powershell
$env:PYTHONNOUSERSITE = "1"
$env:TRAINOMNI_PYTHON = "D:\Codex\TrainOmni\Framework\.venv\Scripts\python.exe"
& .\launch\windows\trainomni.ps1 inspect --task <task.yaml> --allow-local-code
~~~
