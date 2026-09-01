# Multimodal collation contract

ModelIO owns processor semantics and field names. The collator owns only how
per-sample values become a batch; it does not guess a model family from names such
as Qwen, LLaVA or MiniCPM.

The builtin multimodal collator supports deterministic policies per dotted field
path:

| Mode | Result | Typical use |
| --- | --- | --- |
| `stack` | `[batch, ...]`, exact shape required | fixed-size decoded media |
| `pad` | pad tensor dimension 0, preserve trailing shape | text tokens or dense modal tokens |
| `concat` | concatenate dimension 0, preserve trailing shape | flattened image patches and media grids |
| `list` | preserve one value per sample as a tuple | processor/model-owned non-tensor structures |
| `auto` | stack equal shapes; pad variable rank-1 tensors | conservative default |

Paths may be fully qualified, for example
`model_inputs.preference.chosen.input_ids`, or a leaf fallback such as
`pixel_values`. Exact paths take precedence. `field_pad_values` follows the same
lookup rule. Padding side and optional multiple apply to every field explicitly
using `pad` and to rank-1 fields padded by `auto`.

Example for a processor that flattens variable visual patches:

```yaml
data:
  collator:
    module: collator:trainomni/multimodal@1
    config:
      padding_side: right
      pad_to_multiple_of: 8
      field_modes:
        pixel_values: concat
        image_grid_thw: concat
        dense_modal_tokens: pad
      field_pad_values:
        dense_modal_tokens: 0.0
```

Configured modes are strict. `stack` never falls back to padding; `concat` and
`pad` require equal ranks and trailing shapes; unsupported object fields fail
unless explicitly assigned `list`. This prevents a variable number of images or
patches from being silently batched with the wrong layout.

Left padding is supported only when the sample has no explicit position anchors.
Combining `padding_side: left` with `modal_positions`, any `*_positions` field,
`position_ids`, `cache_position`, or `rope_deltas` fails before collation. The v1
collator does not guess how a model-specific anchor should be rebased; such a
model must use right padding or a task-local collator that owns that contract.

Some models need specialized ownership beyond these primitives, such as coupled
offset tensors, packed cross-sample media, or processor-specific Python objects.
Those models register a collator module rather than adding model-name branches to
the builtin collator.
