# Modal branch and fusion contract

A composite model does not hard-code `vision_encoder`, `audio_encoder`, or a
particular VLM class. Its own config declares ordered modal branches:

```yaml
model:
  implementation:
    module: model:trainomni/composite@1
    config:
      branches:
        - name: vision
          modality: image
          input_key: pixel_values
          positions_key: image_positions
          encoder: vision_encoder
          connector: vision_connector
          required: true
        - name: motion
          modality: video
          input_key: video_values
          positions_key: video_positions
          encoder: video_encoder
          connector: video_connector
          required: false
      fusion: fusion_core
      language: decoder
  components:
    vision_encoder: {module: "encoder:example/vision@1"}
    vision_connector: {module: "connector:example/vision@1"}
    video_encoder: {module: "encoder:example/video@1"}
    video_connector: {module: "connector:example/video@1"}
    fusion_core: {module: "fusion:example/cross_modal@1"}
    decoder: {module: "language:example/decoder@1"}
```

Each encoder receives only the tensor or mapping at its `input_key`. Its connector
receives one `ModalFeatures` and returns one `ModalFeatures`. The coordinator then
gives the fusion module an ordered `ModalFeatureSet`; this is the stable boundary
that makes a later audio branch additive rather than a model-core rewrite.

## Contracts

`ModalFeatures` owns one branch's:

- `[batch, modal_tokens, hidden]` embeddings;
- optional `[batch, modal_tokens]` validity mask;
- optional `[batch, modal_tokens]` target text positions;
- optional spatial/temporal grid and branch-local metadata.

`ModalFeatureSet` owns ordered `ModalFeatureBranch` values with unique names and
explicit modality labels. A fusion may inspect branches individually or call
`concatenate()`. Concatenation is deterministic, records branch slices, and
requires compatible batch/hidden dimensions. Positions must exist for every
present branch or none; partial positional alignment fails closed.

The fusion—not the encoder or connector—owns how branches interact with language:
prefix concatenation, token replacement, cross-attention, Q-Former, layered
injection, or another task-local implementation. A task-specific fusion uses the
same generic module loader and hash identity as every other extension.

## Optional and text-only examples

A required branch missing its input fails before fusion. An optional branch may
be absent for an individual batch. If all declared branches are optional and
absent, the fusion receives an empty `ModalFeatureSet`; fusions that support mixed
text-only data must explicitly implement that case. Builtin prefix and
token-replacement fusion pass text through unchanged for an empty set.

Branch input and position keys are unique. Explicit positions returned by an
encoder and positions supplied by ModelIO must be exactly equal. This prevents a
valid-looking feature tensor from being fused at silently different token
positions.

Builtin prefix fusion prepends the validity mask and regenerates ordinary 2-D
`position_ids` for the expanded sequence. It rejects `cache_position`,
`rope_deltas`, higher-rank/model-specific position layouts and other
`*_positions` arguments rather than forwarding stale coordinates. A task-local
fusion must own those model-specific transformations. Token replacement supports
unequal modal-token counts through `ModalFeatures.mask`; padded modal slots must
use the `-1` position sentinel and are never written into text embeddings.

Offline DPO treats media and every non-token-sequence model input as common pair
state. Chosen/rejected prompt prefixes must match exactly, and only the fixed
token-sequence fields may vary by branch. Each branch is additionally bound to its
own full input IDs, true target positions/IDs, reference producer and chosen or
rejected identity before either policy forward.

## Parameter and runtime ownership

Composite components remain directly named PyTorch submodules. Parameter policy,
activation checkpointing and optimizer groups address the component names from
TaskSpec, for example `vision_connector` or `decoder`; they do not depend on the
branch's modality name. Shared encoder/connector components may be referenced by
more than one branch and are registered only once.
