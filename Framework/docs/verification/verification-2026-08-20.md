# Verification Record — 2026-08-20

Public identity verified after the project rename: distribution `trainomni-framework==1.0.0`, import package `trainomni`, CLI `trainomni`, and `trainomni.*` schema/API namespaces. No legacy lowercase package, executable, or schema namespace remains in source/config/tests.

Environment used for optional-runtime verification:

```text
Windows
Python 3.12
torch 2.13.0
transformers 5.15.1
accelerate 1.14.0
peft 0.20.0
pyarrow 23.0.1
```

The temporary virtual environment and model cache are not framework source artifacts.

## 1. Automated suite

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Observed result:

```text
Ran 80 tests
OK
```

Coverage includes canonical positive/negative fixtures, stable fingerprints across processes, plugin trust, config errors, JSON/JSONL/Parquet/TAR readers and importers, exact cursors, mixture/look-ahead resume, accumulation>1 microstep exact resume, real PEFT LoRA train/resume, QLoRA k-bit adapter invocation, `torch.compile`, distributed grouping, DAG/gates/lineage, two-stage physical checkpoint transfer and executor reuse, checkpoint corruption/trust, provenance credential redaction, objective/engine registries, delegated GRPO, pinned VeOmni bridge contracts, configured evaluation placement/inference/autocast, full-parameter P1 optimizer/diagnostic/checkpointing contracts, native cached KD/offline-reference DPO, real torch train/resume/eval/export and static checkpoint probing.

The same 80-test suite on the bundled lightweight Python runtime is expected to skip the optional torch/PEFT and pyarrow runtime cases cleanly.

`tests/test_runtime_data.py::test_data_spec_resume_identity_canonicalizes_only_set_fields` proves that reversed `modalities`/`content_blocks` lists from a prior checkpoint and fresh processes with different Python hash seeds resolve to the same batch-stream identity. Reversing the ordered `datasets` list still fails with `batch stream data_spec mismatch`.

## 2. Public tiny VLM

Model: `Xenova/tiny-random-LlavaForConditionalGeneration`, public Apache-2.0 tiny random LLaVA, approximately 1.05M parameters.

Commands:

```powershell
$plugin = "examples/plugins/tiny_llava.py:PLUGIN"
trainomni --plugin $plugin inspect batch configs/examples/tiny_llava_smoke.yaml --samples 1
trainomni --plugin $plugin train configs/examples/tiny_llava_smoke.yaml --output-dir runs/tiny-llava-smoke
```

Observed encode boundary:

```text
input_ids/labels/attention_mask: [1, 244]
pixel_values: [1, 3, 30, 30]
assistant label span: [242, 244)
text tokens: 244
vision tokens: 225
```

Two real CPU optimizer steps succeeded and produced step 1/2 checkpoints. Resuming from step 1 into a new run produced the same final loss and all 64 model state tensors were `torch.equal` to the uninterrupted step-2 checkpoint.

The trained checkpoint then passed:

- normalized loss evaluation over two image samples；
- HF safe serialization with processor/tokenizer/config；
- `export-manifest.json` generation。

## 3. Single-process exact oracle

`tests/test_torch_e2e.py` runs a small real PyTorch VLM-shaped module and asserts:

```text
train -> step-1/step-2 checkpoint
step-1 exact resume -> step-2
all final model tensors equal
trained checkpoint evaluation succeeds
trained checkpoint export succeeds
```

These tests are part of the 80-test suite when torch is installed and skip cleanly otherwise. The torch e2e module also covers accumulation>1 exact resume, real PEFT LoRA exact resume, `torch.compile`, a two-stage Pipeline with physical checkpoint loading and durable idempotent resume, plus local evaluation model/batch placement under eval/inference/autocast without changing the evaluator/objective request contract.

## 4. Two-process DDP

Windows PyTorch wheels in this environment expose an upstream libuv `torchrun --standalone` rendezvous issue, so the framework-local smoke uses PyTorch file rendezvous without changing DDP execution semantics:

```powershell
python scripts/run_local_ddp_smoke.py `
  --plugin tests/plugins/torch_toy_vlm_plugin.py:PLUGIN `
  --config configs/examples/torch_toy_ddp_resume_smoke.yaml `
  --output-dir runs/torch-toy-ddp-exact
```

Resume into a separate output directory used each rank's step-1 checkpoint. Observed:

```text
rank 0 final tensors equal: True
rank 1 final tensors equal: True
```

Both ranks advanced the same global mixture draw count and produced one optimizer step per global batch group.

## 5. Two-process FSDP2 + DCP

```powershell
python scripts/run_local_ddp_smoke.py `
  --plugin tests/plugins/torch_toy_vlm_plugin.py:PLUGIN `
  --config configs/examples/torch_toy_fsdp2_smoke.yaml `
  --output-dir runs/torch-toy-fsdp2-model
```

Verified:

- `torch.distributed.fsdp.fully_shard` execution on two CPU processes；
- DCP model and optimizer shards；
- two rank-local runtime state sidecars；
- step-1 exact resume to step 2；
- resumed and uninterrupted final loss identical；
- DCP `model_only` load in an uninitialized single process；
- exported state dictionaries have the same 7 keys and every tensor is equal。

The FSDP warning observed concerns the toy module returning a view; it is a conformance warning for the test model and does not invalidate the equality result. Real model plugins should avoid in-place ops on FSDP2 output views.

## 6. Delegated stage

`tests/test_delegated.py` executes a GRPO-shaped `online_rl` stage through the delegated engine. It verifies explicit command authorization, no core model build, redacted request manifest, shell-free subprocess execution, metric collection and physical artifact URI registration.

## 7. VeOmni bridge contract

`tests/test_veomni.py` verifies the VLM-first VeOmni command bridge without pretending the external package ran. It requires an immutable backend revision and exact bridge API, emits a versioned backend contract, executes shell-free, and rejects `resume_level: exact` until real data/RNG/topology conformance exists.

## 8. Static and documentation checks

Final acceptance also runs:

- every Python file through `ast.parse`；
- `pyproject.toml` through `tomllib`；
- every YAML example through `yaml.safe_load` and applicable Pydantic validation；
- Markdown local-link target validation；
- Ruff static lint over source, tests, scripts and examples；
- source-tree scan for generated bytecode/caches。

## 9. Claim boundary

This record proves the framework mechanics and a public VLM integration. It does not claim target-model quality, production GPU throughput, TP/PP/CP/SP/EP performance, real VeOmni/TRL/NeMo/veRL package execution, or Ascend support. Those are separate backend/model conformance matrices rather than missing hidden code paths.

## 10. Full-parameter SFT P1 contract

`tests/test_p1_contracts.py` exercises a true all-component toy step under BF16 AdamW with `foreach=false`. It verifies:

- optimizer-contained trainable numel and component coverage；
- optimizer class/package version/config/actual defaults/state dtype metadata in run and checkpoint manifests；
- vision/connector/language finite nonzero grad norms and exact full-parameter bitwise update scans；
- BF16 representative element unchanged while another component element changes passes with exact count；a completely unchanged BF16 component fails closed；
- separate vision and language activation-checkpoint requests with `use_reentrant=false` and exact receipts；
- uninterrupted versus resumed equality of the entire local registry, including model, optimizer, RNG, data and counters；
- early failure for tampered optimizer identity, ambiguous config, missing/wrong activation receipts and a bitsandbytes request on a non-CUDA device；
- explicit no-fallback wording for AdamW8bit failure。

After the v2 update-evidence change, fresh two-process DDP and FSDP2+DCP runs plus step-1 exact resumes were repeated successfully. DDP rank-local registries and v2 evidence were equal after resume；FSDP2 optimizer metadata and v2 evidence were equal after resume. The optional bitsandbytes implementation was not executed in the CPU-only Framework environment, so target Windows/CUDA conformance remains a P1 runtime gate rather than a Framework claim.

## 11. Offline dense-logit KD contract

`tests/test_cached_kd.py` executes the native `offline-dense-logit-kd` objective and verifies:

- FP32 ground-truth token CE, `T² KL(teacher||student)`, both weighted terms and total against an independent calculation；
- connector gradients pass through a frozen downstream language projection while frozen weights receive no gradients；
- a live teacher model is rejected；
- strict manifest, checkpoint, model/plugin, tokenizer, processor, data, loss-position, BF16 tensor and total-cache identities；
- every external expected identity changes the run fingerprint and fails preflight when it differs from the manifest；
- input/label digest and target alignment checks；BF16 dtype, shape, vocab and prediction-position failures；
- an internally self-consistent manifest/cache with legal but wrong assistant positions is rejected from the real `labels != -100` mask before model forward or optimizer step 1；
- missing or corrupted cache tensor failure before optimizer step 1；
- structured CE/KL/weighted/total metrics plus complete objective identity and latest loss evidence in checkpoints；
- uninterrupted step 2 versus step-1-resumed step 2 equality of the entire state registry and objective identity；
- resume failure when checkpoint objective/cache metadata is tampered。

This is a toy numerical and framework conformance proof. It does not claim that the real 130,560-vocabulary cache has been produced, that the target 16-step GPU run has passed its KL gate, or that KD improves quality. Those remain VLMTrainer acceptance work under the Research contract.

## 12. Offline-reference DPO contract

`tests/test_cached_dpo.py` executes the native `offline-reference-dpo` objective and verifies:

- an independent FP32 oracle for four sequence log-probs, policy/reference ratios, delta, beta-scaled logit, rewards, margin, accuracy and softplus loss；
- chosen and rejected policy forwards both retain gradients through a frozen downstream language layer to the connector；live reference models are rejected；
- strict policy/reference checkpoint, plugin/model, tokenizer, processor, source data, preference manifest, pair direction, common prompt/media model-input and raw FP32 cache identities；
- every external expected identity changes the run fingerprint and fails preflight；invalid beta fails without fallback；
- raw cache corruption and a self-consistent chosen/rejected cache swap fail before model forward or optimizer step 1；
- real labels-mask position derivation, common media equality and identical chosen/rejected preference failures；
- structured DPO metrics, pair/branch-token StateRegistry counters and checkpoint evidence；
- uninterrupted step 2 versus step-1-resumed step 2 full registry equality, plus tampered objective-metadata resume failure。

This proves the native objective/cache/identity/resume mechanics on a toy policy. It does not produce the formal 12-pair/84-token reference cache, run the target 16-step GPU gate, or establish preference quality.
