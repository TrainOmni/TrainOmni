# Attention extension points

Attention is not one setting. TrainOmni keeps four independent change points so a
kernel switch does not silently change model semantics.

| Change | Location | Selection |
| --- | --- | --- |
| eager / SDPA / Flash implementation | runtime/kernels/attention | RunSpec attention_kernel |
| causal / prefix / block / bidirectional semantics | modules/model/attention | TaskSpec model.attention_policy |
| MHA / GQA / MQA / sliding window / RoPE / QKV structure | encoder, language or model module | TaskSpec model module IDs |
| token replacement / modal prefix / cross-attention / Q-Former | modules/model/fusions | TaskSpec fusion module ID |

The runtime kernel service only calls an explicit model
set_attn_implementation boundary and fails if the requested implementation cannot
be applied. It never rewrites an attention mask.

An AttentionPolicy receives input IDs, the current mask and modal positions. It
returns the semantic attention mask plus narrowly scoped model kwargs. The builtin
model_default policy validates a normal two-dimensional padding mask and otherwise
leaves causal semantics to the language architecture. New prefix/block semantics
are ordinary custom attention_policy modules and must declare a compatible model
capability.

Changing MHA to GQA, RoPE behavior, windowing or QKV layout changes checkpoint
structure and therefore belongs in the concrete encoder/language/model module.
Changing where modality features enter the decoder belongs in Fusion. Neither is
implemented as a trainer flag.

All four can be extended with the generic module mechanism described in
extensions.md. Only the runtime kernel is RunSpec execution policy; the other
three alter TaskSpec semantics and therefore change the task digest.
