# Core correctness closure — 2026-09-01

This document records the corrected Framework contracts. It is implementation
evidence, not a model-quality or Linux/NCCL/Ascend performance claim.

| Area | Corrected contract | Automated evidence |
| --- | --- | --- |
| Builtin provenance | Framework version, release provenance token and newline-normalized installed Python source-tree SHA-256 enter every module lock; pre-fix framework checkpoints are rejected | `test_builtin_provenance_changes_module_lock`, `test_checkpoint_from_pre_fix_framework_version_is_not_exact_resumable` |
| Effective-batch loss | Objectives expose numerators/denominators; the engine backpropagates local numerators and applies one globally summed denominator after DDP/FSDP synchronization | two-process Gloo oracle in `test_two_rank_global_loss_and_primary_failures_are_coordinated` |
| Objective metrics | every metric declares `sum` or `weighted_mean` with numerator/denominator; accumulation and ranks sum declared state before finalization | unequal pair-count/accuracy/delta two-rank Gloo oracle plus evaluator tests |
| Checkpoint relocation | Run/task/module semantic receipts exclude physical checkpoint, columnar staging and manifest-pinned local Transformers paths; physical run location is replaceable metadata | `test_full_resume_allows_checkpoint_directory_relocation_in_the_same_run_root`, task/source relocation tests |
| Finite input | EOF flushes packer state; `drop_last=false` returns the partial tail, `drop_last=true` records and drops it | finite-pipeline and sequence-packer flush tests |
| Multi-rank exhaustion | finite or unknown-length sources fail before rank sharding; only explicitly repeating sources can enter the current multi-rank loop | `test_multi_rank_unknown_or_finite_exhaustion_fails_closed` |
| Operation assembly | train/evaluate/export construct only their required data path; model-only evaluation/export do not touch a missing training source | `test_evaluate_and_export_do_not_construct_the_training_source` |
| Checkpoint failures | pure rank-local scheduler/objective/stream/scaler/RNG capture completes an all-rank outcome phase before gather/I/O; rank zero broadcasts filesystem and identity-materialization failures | rank-one injected state capture and rank-zero filesystem failure two-rank tests |
| Local-rank device binding | CUDA/NPU local rank is selected before process-group initialization and therefore before FSDP device-mesh creation | device binding/order tests |
| Distributed model-only evaluation | single-process load restores a distributed objective only when all saved rank states are exactly equal | rank-invariant positive and rank-dependent negative checkpoint tests |
| DPO pair alignment | chosen/rejected logical prompt tokens/mask, each branch's own logical prompt values for position/type/cache fields, and all common non-sequence inputs/media match; response length and physical left-padding offset may differ, while non-contiguous masks fail | unequal-response left-padding positive plus token/attention/position/type/cache/media/branch pre-forward negatives |
| Prefix fusion | expanded attention/ordinary 2-D positions are regenerated; stale cache/rope/model-specific position fields fail closed | prefix position/mask/cache/rope tests |
| Token replacement | modal validity masks permit unequal counts; padded slots require the `-1` sentinel and are not written | unequal-modal-count positive and sentinel-negative tests |
| Transformers assets | immutable revision identifies only a remote repository snapshot; an existing/local-files-only asset requires a producer manifest for checkpointing and otherwise retains physical best-effort identity | real temporary local-directory, remote-revision, relocation and checkpoint rejection tests |
| Parquet/Arrow snapshots | only a valid producer manifest makes paths relocatable; unpinned diagnostics retain declared/resolved paths and cannot checkpoint | pinned relocation/changed-manifest and unpinned A/B collision tests |
| Offline KD/DPO caches | schema-v4 additionally binds the complete uncollated model-input mapping; builtin supervision hashes current media/auxiliary inputs before collation, while schemas v2/v3, stale media/position inputs and left/right-padding collisions fail before forward | cache schema, numeric oracle and KD/DPO padding/alignment/media/auxiliary negatives |
| Documentation/catalog | package version, test evidence, builtin descriptor count and support matrix refer to the same corrected tree | wheel import/CLI/catalog smoke plus full suite |

## External identity trust boundary

An asset or dataset manifest is intentionally small and producer-owned. Its
digest avoids hashing multi-gigabyte weights or tens of millions of data rows on
every launch. The manifest must itself list immutable object versions or file
hashes. Framework verifies and persists the declared digest as identity; it does
not prove that a dishonest producer described the payload truthfully.

For cached KD/DPO, the producer identity digest must combine the precise producer
model state, processor, tokenizer and generation contract. Per-sample bindings
then prevent cache reuse against different expanded tokens, attention/padding
layout, absolute target masks, media tensors, auxiliary model inputs or pair
branches before policy forward. The current complete-input digest is computed by
Framework supervision after ModelIO and before collation rather than being copied
from the cache manifest; this preserves per-sample identity when a batch later
pads text or concatenates variable media tensors.

## Hardware boundary

The current host executed the complete source suite and an actual two-process
CPU/Gloo DDP control/numerical/failure gate. Existing single-GPU CUDA VLM evidence
remains historical pre-fix route evidence, not validation of the current corrected
tree; current-tree real-VLM revalidation is pending. New framework provenance
deliberately prevents exact resume across the code change.
Real multi-rank CUDA/NCCL, FSDP2 sharding at world size greater than one, Linux
DeepSpeed, Ascend/torch_npu/HCCL and remote object-store transport remain server
deployment gates.
