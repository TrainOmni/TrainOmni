# Padding-free / variable-length training

This is an **opt-in model integration**, not a global attention toggle. It is
separate from `sequence@1`, which retains its existing padded/dense-mask behavior.

## Data interface

```yaml
packer:
  module: packer:trainomni/padding_free@1
  config:
    max_length: 4096
    pad_token_id: 0
    max_samples_per_pack: 8
    concat_fields: [pixel_values, image_grid_thw, image_counts]
collator:
  module: collator:trainomni/padding_free@1
  config:
    field_modes:
      pixel_values: concat
      image_grid_thw: concat
      image_counts: concat
```

`max_length` is the input-token packing budget, **not** a promise that visual
expansion fits within that budget. The model/processor owner must enforce its
expanded context limit. `pad_token_id` is retained from the common field-routing
config but no sequence padding is emitted. A pack contains multiple source
samples; RunSpec `per_device_batch_size` must be **1 pack**. More than one pack
per collation, left padding, padding field modes and pad-to-multiple are rejected.

Uncollated tensors include:

- `input_ids`, `attention_mask`, `position_ids`, `packed_segment_ids`: `[T]`;
- `packed_cu_seqlens`: int32 `[N+1]`, starts at zero and ends at `T`;
- labels/supervision remain token aligned; each appended sample's first label
  is ignored, so the previous sample cannot predict its first token;
- `packed_lengths` supervision records individual lengths;
- explicit modal concat/offset/list policies are inherited from sequence packing.

`T` is the actual token sum, including EOF tails, never `max_length` padding.
The collator adds a leading singleton batch axis to the text and cumulative
length tensors. Neither component allocates a `[T,T]` mask. Pending pack buffers
use the same state contract as ordinary packing. These modules deliberately do
not satisfy the old dense-mask packer/collator capability chain.

## Model / fusion interface

The model descriptor must provide `model.attention.padding_free`; existing
packed-mask support alone is insufficient and preflight rejects that pairing.
Only advertise the capability after implementing and verifying consumption.

```python
from trainomni.runtime.kernels.attention.varlen import (
    VarlenLayout, padding_free_forward,
)

text_layout = VarlenLayout.from_packed(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    segment_ids=packed_segment_ids,
    cu_seqlens=packed_cu_seqlens,
)
# Fuse images independently within each text_layout segment. For visual-prefix
# insertion, expanded_lengths must include EACH sample's actual visual tokens.
expanded_layout = VarlenLayout(tuple(expanded_lengths))
output = padding_free_forward(
    language_model,
    inputs_embeds=concatenated_embeddings,  # [1, sum(expanded_lengths), hidden]
    layout=expanded_layout,
)
# Align output logits back to the text targets according to the fusion mapping.
```

Validation of text boundaries, segment IDs, position resets and all-valid masks
happens before model work. The language call constructs position IDs and the
upstream block-diagonal causal bias from **post-fusion** lengths. Reusing text
lengths after visual insertion is incorrect and a token-count mismatch fails.

The helper uses Transformers' registered attention/mask interfaces, not patched
model layers. It selects `trainomni_varlen_cutlass` only on the supplied language
model; the vision tower keeps its own attention. Keep the global RunSpec kernel
at `auto` for such an explicitly configured plugin. Record the selected backend
and tested dependency versions in the task-local model config/environment lock.
Later direct Llama forwards without the layout fail closed; for an independent
ordinary forward, explicitly switch the language model back to `sdpa`/`eager`.

## Kernel and support boundary

The optional backend is **xFormers CUTLASS**, using upstream
`memory_efficient_attention` with an explicit `BlockDiagonalCausalMask` and
explicit forward/backward operator selection. It is **not FlashAttention**.
There is no silent SDPA/dense-mask/CPU fallback. xFormers is lazy-imported and is
not a mandatory base dependency. Do not install a binary that replaces the
project Torch just to satisfy a generic dependency extra: use a compatible
platform build and rerun the CUDA tests below.

Current verification: LlamaForCausalLM, causal self-attention, CUDA BF16/FP16,
zero attention dropout, batch-one concatenated sequences; real VLM evidence is
BF16 connector training. GQA repeats K/V heads explicitly to preserve backward
support (extra linear memory/copy cost); it does not claim a specialized fused
GQA training kernel. An attention bias stores sequence boundaries, not a dense
tensor. No additional CUDA kernel is implemented in TrainOmni.

Not claimed: arbitrary VLMs/mRoPE, FlashAttention-2/3/4, KV-cache generation,
sliding-window/softcap attention, dropout, compile/CUDA graphs, distributed
execution, exact training resume on this new backend, or throughput improvement.
The tested metadata buffer restore is not a full-training resume claim.

Upstream interfaces: [xFormers operators and structured attention biases](https://facebookresearch.github.io/xformers/components/ops.html),
[Transformers attention registration](https://huggingface.co/docs/transformers/main/en/attention_interface).

See [local CUDA verification](../verification/padding-free-20260903.md).
