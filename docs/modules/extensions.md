# Module extension contract

Custom Objective is one example of the extension system, not a special case.
Every code-bearing variation point uses the same ModuleDescriptor, ModuleId,
typed config, capability negotiation and source identity rules.

## Extension kinds

| Kind | Owns |
| --- | --- |
| data_source | storage access and resumable source cursor |
| data_adapter | physical row/schema conversion into `OmniSample` |
| sample_transform | media loading, normalization, filtering, augmentation |
| model_io | processor, tokenizer, template, special tokens and tensor mapping |
| supervision | labels, masks, target positions and preference branches |
| packer | sequence packing and buffered state |
| collator | padding and final OmniBatch construction |
| encoder | image/video and later audio feature production |
| connector | modal feature projection and resampling |
| fusion | token replacement, prefix, cross-attention and layered injection |
| language | embedding/decoder/output-head boundary |
| model | monolithic model or composite coordinator |
| attention_policy | semantic attention mask and position rules |
| objective | forward plan, loss numerics, reductions and metrics |
| parameter_policy | freeze/full/component/adapter selection |
| evaluator | held-out metric state and reduction |
| exporter | deployable artifact conversion |

Optimizer, scheduler, precision, attention kernel, compilation and distributed
topology are execution services selected by RunSpec. They are not semantic task
modules.

## Builtin or task-local

Reusable, generally supported behavior is a builtin under
src/trainomni/modules/<category>/<name>/. It is part of the single TrainOmni Python
distribution; one module does not mean one wheel or one environment.

Experimental or task-specific behavior can be kept next to the task:

~~~
<task-root>/modules/<category>/<name>/
├── __init__.py
├── config.py
├── module.py
└── module.toml
~~~

The task lists module ID, relative directory and exact source-tree SHA-256 under
local_modules. The consumer then references the same module ID at the normal
TaskSpec field. The loader is generic: it does not branch on Objective and does not
care whether the descriptor is for an encoder, fusion, loss, data adapter or any
other registered kind.

Before executing local code, Framework requires explicit opt-in, verifies the
directory stays below the task root, rejects symlinks, checks every file against
the pinned digest, validates manifest ID/API/entrypoint and registers the returned
descriptor in a task-scoped registry. It uses a digest-derived import namespace
and never changes sys.path.

This provides module namespace and provenance isolation. It is not a Python
security sandbox: local modules are trusted code.

## Isolation rules

- A module receives only its own typed config and a narrow construction context.
- A module never receives the flattened TaskSpec or RunSpec.
- Cross-module values use contracts from trainomni.contracts.
- Encoders/connectors exchange one `ModalFeatures`; a composite coordinator gives
  fusion an ordered `ModalFeatureSet`, so multiple image/video branches and a
  later audio branch do not require core model-name branches.
- A concrete implementation can import its category protocol and shared helpers,
  but cannot import a sibling implementation.
- Capability mismatch fails before any factory constructs a model or data stream.
- Stateful modules must make every mutable cursor/buffer part of state_dict and
  load_state_dict.
- The task digest and module source digest enter checkpoint identity, so changed
  code cannot silently resume old state.

Loss-specific details and a custom Objective example are in
custom-objective.md. The same directory/manifest/descriptor process applies to all
module kinds listed above.

The composite modal routing and custom-fusion boundary are specified in
`../contracts/modal-features.md`.
