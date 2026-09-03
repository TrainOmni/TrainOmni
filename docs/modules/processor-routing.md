# Processor fields and composite model input routing

For a processor compatible with your tokenizer/chat template, builtin
`model_io:trainomni/transformers@1` supports explicit field ownership:

```yaml
model_io:
  module: model_io:trainomni/transformers@1
  config:
    processor_name_or_path: /path/to/compatible/processor
    local_files_only: true
    asset_manifest_sha256: <producer-verified-asset-manifest-sha256>
    modal_token_id: 12345  # replace with THIS tokenizer's actual placeholder ID
    unmapped_fields: error
    field_routes:
      input_ids: input_ids
      attention_mask: attention_mask
      pixel_values: vision.hidden_states
      image_grid_thw: vision.grid_thw
    discard_fields: [mm_token_type_ids]
```

`modal_token_id` makes ModelIO derive int64 `modal_positions` from the actual
input IDs and advertise `batch.modal_positions`. It also masks those placeholders
from labels. Without the option the capability is not advertised. This is not a
hardcoded Qwen token ID or an assumption about a 2x2 merger. The model-specific
producer must expand the correct visual placeholders and preserve image grouping;
the connector/fusion validate output counts. Unknown routed/discarded ownership
fails when `unmapped_fields: error`; the backward-compatible default is `keep`.

`field_routes` accepts dotted target paths. Renaming/dropping root `input_ids`,
overlapping destinations, or assigning one field to both routing and discard
is rejected. Existing explicit `batch_axis_fields` semantics are preserved;
singleton image/grid axes are not blindly squeezed.

A Qwen vision tower plus a **different** LLM tokenizer is not a drop-in Qwen
processor. That combination needs a task-local tokenization adapter. The runnable
Qwen3.5 raw-ViT/merger/MiniCPM5 example reuses the builtin public
`TransformersModelIO.normalize_encoded(...)` boundary for routing, positions and
assistant loss masks, while the task owns cross-tokenizer chat/image expansion.
It does not silently feed the Qwen tokenizer's IDs into MiniCPM5.

Prefer explicit ownership to filtering LLM keyword arguments by signature:
silent filtering can hide lost visual information. Pack the routed nested leaves
using the [packing contract](../contracts/sequence-packing.md).
