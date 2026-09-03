# Sequence packing and attention isolation

TrainOmni does not call token concatenation “packing” unless it also preserves
causal sample isolation and exact resume.

The builtin `packer:trainomni/sequence@1` uses a resumable sequential buffer and
emits fixed-length examples. For every emitted pack it:

- concatenates token IDs and resets `position_ids` per source sample;
- masks the first label of every appended sample, preventing a cross-sample next
  token target;
- pads labels with the configured ignore index;
- emits validity and segment IDs;
- emits a verified lower-triangular block-diagonal boolean attention mask;
- applies explicit policies to token-aligned, concatenated modal, offset-position,
  and list-valued fields;
- serializes the complete pending buffer and token cursor for exact resume.

Unconfigured model-input fields fail closed. A tensor cannot be guessed as text,
flattened image patches, modal positions, or a processor-owned object.

Example field ownership:

```yaml
packer:
  module: packer:trainomni/sequence@1
  config:
    max_length: 4096
    pad_token_id: 0
    sequence_fields: [token_type_ids]
    concat_fields: [vision.hidden_states, vision.grid_thw, vision.image_counts]
    offset_fields: [modal_positions]
```

Nested fields use explicit dotted leaf paths. The packer flattens paths for
policy lookup and reconstructs the same nested mapping afterward. It does not
guess an axis for an entire vision subtree: flattened patches, grid rows,
per-example image counts and token positions have different meanings.
Overlapping paths, unknown fields and missing declared fields fail closed.
An explicitly listed `list_fields` subtree is preserved as a list, not tensor
concatenated; its consumer must implement that contract.

Per uncollated pack, `packed_attention_mask` is `[1,S,S]`: **1 is the head axis**,
not a batch axis. Collation stacks packs to `[B,1,S,S]`. Do not unsqueeze again.
The attention policy reports actual shape/dtype/device on invalid inputs.

Packing additionally requires an attention policy providing
`model.attention.packed`. The builtin packed policy re-derives the mask from token
validity and segment IDs before model forward and rejects corruption. It can emit
boolean or FP32 additive 4D masks. It also requires a sequence-length-preserving
fusion such as token replacement; prefix fusion cannot silently use the same mask
because it inserts modal tokens.

For `B > 1`, concatenated pixels alone are insufficient. The task encoder must
retain image grid and per-pack grouping in `ModalFeatures.grid/metadata`; the
connector must return padded `[B,M_max,H]`, a matching mask, and positions whose
padding is `-1`. One working explicit layout is
`vision.image_counts[B,max_examples_per_pack]`, with **zero** count padding:

```yaml
field_modes:
  model_inputs.vision.hidden_states: concat
  model_inputs.vision.grid_thw: concat
  model_inputs.vision.image_counts: pad
  model_inputs.modal_positions: pad
field_pad_values:
  model_inputs.vision.image_counts: 0
```

No new connector ABI is required: `ModalFeatures.metadata` already carries the
boundaries. There is no blanket batch-size-one restriction on dense packing.
The separate padding-free collator still requires one **pack**, which can
contain multiple samples. The Transformers language adapter converts additive
masks to the embedding dtype while keeping finite sentinels finite; boolean
masks are unchanged.

Models that use FlashAttention variable-length metadata, document masks, or a
different packed-attention representation register an attention policy and,
where necessary, a packer module. The training loop remains unchanged.

For the separate opt-in unpadded representation and tested upstream CUTLASS
varlen path, see [padding-free training](padding-free.md). Ordinary
`sequence@1` does not turn into FlashAttention or padding-free execution merely
by selecting another runtime kernel.
