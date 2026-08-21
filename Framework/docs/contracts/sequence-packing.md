# Sequence packing and attention isolation

TrainOmni does not call token concatenation “packing” unless it also preserves
causal sample isolation and exact resume.

The builtin `packer:trainomni/sequence@1` uses a resumable first-fit buffer and
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
    concat_fields: [pixel_values, image_grid_thw]
    offset_fields: [modal_positions]
```

Packing additionally requires an attention policy providing
`model.attention.packed`. The builtin packed policy re-derives the mask from token
validity and segment IDs before model forward and rejects corruption. It can emit
boolean or FP32 additive 4D masks. It also requires a sequence-length-preserving
fusion such as token replacement; prefix fusion cannot silently use the same mask
because it inserts modal tokens.

Models that use FlashAttention variable-length metadata, document masks, or a
different packed-attention representation register an attention policy and,
where necessary, a packer module. The training loop remains unchanged.
