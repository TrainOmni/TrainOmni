# TrainOmni real VLM five-stage validation

This directory is a disposable consumer of `Framework`, not framework source.
It validates one uninterrupted artifact chain on the local Qwen3.5 vision and
MiniCPM5-1B checkpoints:

1. connector alignment;
2. multimodal pretraining;
3. supervised fine-tuning;
4. offline dense-logit knowledge distillation;
5. offline-reference DPO;
6. final evaluation, export, and reload.

Task-local model/data adapters live under `modules/`. Generated checkpoints,
caches, receipts, and exports live under `runs/` and `cache/` and are excluded
from the Framework package. The data is deliberately tiny: this proves the
training paths and artifact handoff, not model quality.

Status: complete. The accepted two-step uninterrupted runs are
`01-alignment-v2`, `02-pretraining-v3`, `03-sft`, `04-kd`, and `05-dpo-v2`.
Follow-up accepted runs are `06-lora-sft`, `07-lora-dpo`, and
`08-batching-stress`; they cover strict adapter export/reload, batch-size 2,
unequal text/image shapes, one/two images per sample, and gradient accumulation 2.
Fresh-process exact-resume receipts are under `resume-validation/` for full SFT,
pretraining, alignment, offline KD and offline DPO. The reusable
`verify_real_exact_resume.py` verifier compares four uninterrupted steps with two
steps plus resume to step four, then runs held-out evaluation. Large comparison
checkpoints are temporary; receipts retain logical model/optimizer/runtime digests.
The `extension-*.task.json` tasks and `verify_extension_routes.py` independently
exercise a task-local custom Objective, eager/SDPA attention selection, weighted
multi-source sampling, real multimodal sequence packing and ordered video-frame
input. Their six compact receipts are under `extension-validation/`; their large
checkpoints are also temporary.
Earlier sibling run directories are retained failure evidence from defects found
during the gate. The final complete artifact is `artifacts/stage-05-final`; large
intermediate payloads were pruned after downstream consumption, while manifests,
metrics, resolved identities and evaluation receipts remain.

The distributed gate adds direct PyTorch DDP and FSDP2 world-size-one execution
on the RTX 4060 Ti, including real process groups, forward/backward/update,
portable checkpoint, resume and held-out evaluation. These are backend probes,
not multi-card claims; Linux/NCCL multi-rank remains a server gate.

The medium-data gate deterministically converts the pinned diagram and InterGPS
Parquet bundle into 240/30 and 1024/128 train/validation splits, with bounded
64/16 KD and DPO cache subsets. Seven routes complete 16 distinct-data optimizer
steps each. Reproducible preparation/summarization scripts are in this directory;
the compact authoritative result is `medium-validation/receipt.json`. Generated
media, caches, checkpoints and run directories are intentionally Git-ignored.
