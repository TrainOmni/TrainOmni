# Canonical sample and conversation contract

`OmniSample` has two mutually exclusive semantic forms:

1. ordered flat `content` blocks for pretraining/interleaved-content tasks;
2. ordered role-aware `messages` for instruction/SFT chat tasks.

Every block is one of text, image, video, or audio and carries optional metadata.
A message owns a non-empty role, ordered content blocks, and optional metadata.
The canonical source never stores processor tensors or model-specific image-token
IDs.

JSONL conversation example:

```json
{"sample_id":"s1","messages":[{"role":"user","content":[{"kind":"image","value":"images/a.png","metadata":{"sha256":"..."}},{"kind":"text","value":"What is shown?"}]},{"role":"assistant","content":[{"kind":"text","value":"A geometric diagram."}]}]}
```

Media resolution and image decoding traverse both forms without flattening role
or block order. A sample cannot provide both `content` and `messages`, preventing
ambiguous template semantics.

## Transformers chat-template adapter

For message samples, the builtin Transformers ModelIO calls the selected
processor's `apply_chat_template` with tokenization, tensor output, and assistant
token-mask output enabled. It converts canonical blocks to the standard
role/content structure and moves the returned assistant mask from model inputs to
the supervision contract.

`conversation_mode` is explicit:

- `auto`: chat-template messages; use direct processor encoding for flat content;
- `required`: reject flat samples;
- `disabled`: reject message samples.

By default, a message sample must produce an assistant token mask aligned exactly
with one-dimensional `input_ids`. Missing or ambiguous mask fields fail before
collation/model forward, rather than silently applying language-model loss to
system and user prompt tokens. A template without assistant generation markers
therefore cannot be used for SFT accidentally.

`batch_axis_fields` explicitly lists processor outputs whose leading axis is the
single-sample batch axis. Defaults are `input_ids`, `attention_mask`,
`token_type_ids`, and `mm_token_type_ids`; a present listed field must be a
tensor shaped `[1, ...]` with at least two dimensions. Only that axis is removed.
All other outputs retain their exact shape. In particular, `image_grid_thw`,
`video_grid_thw`, flattened patches and video tensors do not lose an image,
patch or frame axis just because its size happens to be one. Their subsequent
`concat`/`stack`/`pad` behavior belongs to the collator configuration.

For a processor that actually returns `pixel_values` as `[batch, C, H, W]`, set
`batch_axis_fields: [input_ids, attention_mask, pixel_values]` and use `stack`
for images. Do not use that setting for Qwen-style flattened patch tensors.
This list replaces the defaults; include any other token fields emitted by the
chosen processor. Non-leading or model-specific batch axes require a task-local
ModelIO, not automatic shape inference.

Processors with a different conversation schema, multi-turn loss policy, tool
semantics, or field layout use a ModelIO module. The generic extension mechanism
keeps that variation outside DataSource, Objective, and the training loop.
