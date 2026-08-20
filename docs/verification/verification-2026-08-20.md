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
Ran 57 tests
OK
```

Coverage includes canonical positive/negative fixtures, stable fingerprints across processes, plugin trust, config errors, JSON/JSONL/Parquet/TAR readers and importers, exact cursors, mixture/look-ahead resume, distributed grouping, DAG/gates/lineage, two-stage physical checkpoint transfer and executor reuse, checkpoint corruption/trust, provenance credential redaction, objective/engine registries, delegated GRPO, pinned VeOmni bridge contracts, real torch train/resume/eval/export and static checkpoint probing.

The same 57-test suite on the bundled lightweight Python runtime passes with three expected skips: two torch end-to-end tests and one pyarrow reader test.

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

These tests are part of the 57-test suite when torch is installed and skip cleanly otherwise. The second torch e2e test executes a two-stage Pipeline, loads the first stage's physical checkpoint in the second stage, then reuses the same executor for an idempotent durable-state resume.

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
