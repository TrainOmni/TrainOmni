# Core correctness closure — 2026-09-01

This document records the corrected Framework contracts. It is implementation
evidence, not a model-quality or Linux/NCCL/Ascend performance claim.

| Area | Corrected contract | Automated evidence |
| --- | --- | --- |
| Builtin provenance | Framework version, release provenance token and newline-normalized installed Python source-tree SHA-256 enter every module lock; pre-fix framework checkpoints are rejected | `test_builtin_provenance_changes_module_lock`, `test_checkpoint_from_pre_fix_framework_version_is_not_exact_resumable` |
| Effective-batch loss | Objectives expose numerators/denominators; the engine backpropagates local numerators and applies one globally summed denominator after DDP/FSDP synchronization | two-process Gloo oracle in `test_two_rank_global_loss_and_primary_failures_are_coordinated` |
| Checkpoint relocation | Run/task/module semantic receipts exclude physical checkpoint, columnar staging and manifest-pinned local Transformers paths; physical run location is replaceable metadata | `test_full_resume_allows_checkpoint_directory_relocation_in_the_same_run_root`, task/source relocation tests |
| Finite input | EOF flushes packer state; `drop_last=false` returns the partial tail, `drop_last=true` records and drops it | finite-pipeline and sequence-packer flush tests |
| Multi-rank exhaustion | finite or unknown-length sources fail before rank sharding; only explicitly repeating sources can enter the current multi-rank loop | `test_multi_rank_unknown_or_finite_exhaustion_fails_closed` |
| Operation assembly | train/evaluate/export construct only their required data path; model-only evaluation/export do not touch a missing training source | `test_evaluate_and_export_do_not_construct_the_training_source` |
| Primary filesystem errors | rank zero broadcasts checkpoint and identity-materialization success/failure, without a success-only barrier | two-rank checkpoint failure plus direct `build_engine` failure tests |
| Local-rank device binding | CUDA/NPU local rank is selected before process-group initialization and therefore before FSDP device-mesh creation | device binding/order tests |
| Distributed model-only evaluation | single-process load restores a distributed objective only when all saved rank states are exactly equal | rank-invariant positive and rank-dependent negative checkpoint tests |
| DPO pair alignment | chosen/rejected prompt tokens and all common non-sequence inputs, including media, must match; only a fixed token-sequence field set may vary | prompt/media/branch corruption and config-bypass tests |
| Prefix fusion | expanded attention/ordinary 2-D positions are regenerated; stale cache/rope/model-specific position fields fail closed | prefix position/mask/cache/rope tests |
| Token replacement | modal validity masks permit unequal counts; padded slots require the `-1` sentinel and are not written | unequal-modal-count positive and sentinel-negative tests |
| Transformers assets | remote commit revision or producer-owned local manifest digest is mandatory for checkpointed reproducibility; local staging roots may move | asset provenance, relocation and unpinned-exact-resume tests |
| Parquet/Arrow snapshots | producer manifest plus logical fragment layout is semantic identity; physical paths/mtime/size are transport metadata; changed manifest fails | columnar task/source relocation and changed-manifest tests |
| Offline KD/DPO caches | schema-v2 cache binds sample-selected tensor shard, expanded input IDs, true supervised positions, target IDs, producer model/processor/tokenizer identity and teacher/chosen/rejected branch | cache schema, numeric oracle and pre-forward corruption tests |
| Documentation/catalog | package version, test evidence, builtin descriptor count and support matrix refer to the same corrected tree | wheel import/CLI/catalog smoke plus full suite |

## External identity trust boundary

An asset or dataset manifest is intentionally small and producer-owned. Its
digest avoids hashing multi-gigabyte weights or tens of millions of data rows on
every launch. The manifest must itself list immutable object versions or file
hashes. Framework verifies and persists the declared digest as identity; it does
not prove that a dishonest producer described the payload truthfully.

For cached KD/DPO, the producer identity digest must combine the precise producer
model state, processor, tokenizer and generation contract. Per-sample bindings
then prevent cache reuse against different expanded tokens, target masks or pair
branches before policy forward.

## Hardware boundary

The current host executed the complete source suite and an actual two-process
CPU/Gloo DDP control/numerical gate. Existing single-GPU CUDA VLM evidence remains
valid as historical execution evidence for the pre-fix routes, but the new
framework provenance deliberately prevents exact resume across the code change.
Real multi-rank CUDA/NCCL, FSDP2 sharding at world size greater than one, Linux
DeepSpeed, Ascend/torch_npu/HCCL and remote object-store transport remain server
deployment gates.
