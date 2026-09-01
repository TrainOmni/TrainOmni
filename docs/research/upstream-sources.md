# Upstream source reference ledger

The upstream repositories are pinned source references, not vendored source or
Git submodules. Most are reading material only; PyTorch/Transformers are base
dependencies and DeepSpeed has a separately selected optional thin adapter.
Physical checkouts are deliberately outside the Framework tree. On the current
development machine they are under:

```text
D:\Codex\TrainOmniTemp\framework-upstream-references-20260821\upstreams
```

The location is disposable and is not used by imports, tests, builds, or runtime.
The reproducible identities used during the redesign are:

| Source | Commit | Runtime status | Primary reference area |
| --- | --- | --- | --- |
| Transformers | `0c92811846095910816a87aca50050d10c545270` | base dependency | model/processor and attention contracts |
| VeOmni | `b89258944a9e2a04556b9e832b21d096f31def53` | reference only | multimodal data and distributed recipes |
| TRL | `88b99c2ce4adaeaf449304e9d95f9b52a759bd8b` | reference only | preference/KD numerics and tests |
| PEFT | `b60f0552e7e51a671849665ab834401cb748ec40` | reference only | LoRA injection and artifact conventions |
| Accelerate | `2583573aa6b9b205b82e6d8aeaba446b6bb771ab` | reference only | device/mixed-precision lifecycle edge cases |
| TorchTitan | `00cffaeb33bffff37abd1bd9b50773b950b0dad3` | reference only | FSDP2/checkpointing patterns |
| NeMo AutoModel | `c63a6633f48fd68a4ff738c63ce05389446533ec` | reference only | VLM composition constraints |
| ms-swift | `a54a4ae5c8680451ba4ddc91ad4577a38c74d560` | reference only | VLM templates/masks/model metadata |
| LLaMA-Factory | `c4e09c7cbe18844816af9e18a97fe465515edbcd` | reference only | task/data configuration UX |
| DeepSpeed | `cf44300453eb0af79ed84ed8f1cb49d57478bd76` | optional Linux backend + reference | ZeRO execution/config/checkpoint boundary |
| VLMEvalKit | `e8e78f05f3080fe28154f2130321f17951c3be94` | external evaluation backend | VLM benchmark inference, scoring, and result formats |

No code has been copied from these repositories into the replacement
implementation. Optional DeepSpeed is imported through its public API only. Any
future copied or substantially derived code requires a license check and a
source/commit/path entry in a third-party notice.

VLMEvalKit is installed editable from its external checkout and is not a Core
dependency. The current Windows host applies a recorded compatibility overlay
from `D:\Codex\TrainOmni\FrameworkValidation\evaluation\patches`; the upstream
checkout remains the authoritative source and no project fork is maintained.
