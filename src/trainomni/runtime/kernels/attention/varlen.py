"""Opt-in Llama padding-free training via upstream xFormers CUTLASS.

This is not FlashAttention and has no silent backend fallback. Packing/fusion
owners pass final (post-visual-expansion) lengths; neither kernels nor model
layers are reimplemented here. No dense block mask is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from trainomni.core.errors import SpecError
from trainomni.modules.data._tensors import binary_mask

BACKEND = "trainomni_varlen_cutlass"


@dataclass(frozen=True, slots=True)
class VarlenLayout:
    lengths: tuple[int, ...]

    def __post_init__(self):
        lengths = tuple(self.lengths)
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
            raise SpecError("varlen lengths must be nonempty positive integers")
        if sum(lengths) > torch.iinfo(torch.int32).max:
            raise SpecError("varlen total length exceeds int32 capacity")
        object.__setattr__(self, "lengths", lengths)

    @property
    def total_tokens(self):
        return sum(self.lengths)

    def position_ids(self, device):
        return torch.cat([torch.arange(n, device=device) for n in self.lengths])[None]

    @classmethod
    def from_packed(cls, *, input_ids, attention_mask, position_ids, segment_ids, cu_seqlens):
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise SpecError("padding-free input_ids must be [1, total_tokens]")
        if input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise SpecError("padding-free input_ids must be [1, total_tokens]")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise SpecError("padding-free input_ids must be int32/int64")
        for name, tensor in (("position_ids", position_ids), ("segment_ids", segment_ids)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != input_ids.shape
                or tensor.dtype not in {torch.int32, torch.int64}
                or tensor.device != input_ids.device
            ):
                raise SpecError(f"padding-free {name} must align with input_ids")
        valid = binary_mask(
            attention_mask, field="padding-free attention_mask", shape=input_ids.shape
        )
        if valid.device != input_ids.device or not bool(valid.all().item()):
            raise SpecError("padding-free attention_mask cannot contain padding")
        if (
            not isinstance(cu_seqlens, torch.Tensor)
            or cu_seqlens.ndim != 2
            or cu_seqlens.shape[0] != 1
            or cu_seqlens.shape[1] < 2
            or cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != input_ids.device
        ):
            raise SpecError("packed_cu_seqlens must be int32 [1, sequences+1] on input device")
        offsets = cu_seqlens[0].tolist()
        if offsets[0] != 0 or offsets[-1] != input_ids.shape[1]:
            raise SpecError("packed_cu_seqlens endpoints do not match input_ids")
        layout = cls(tuple(b - a for a, b in pairwise(offsets)))
        if not torch.equal(position_ids, layout.position_ids(input_ids.device)):
            raise SpecError("position_ids do not reset at packed_cu_seqlens boundaries")
        expected_segments = torch.repeat_interleave(
            torch.arange(len(layout.lengths), device=input_ids.device),
            torch.tensor(layout.lengths, device=input_ids.device),
        )[None]
        if not torch.equal(segment_ids, expected_segments):
            raise SpecError("segment_ids disagree with packed_cu_seqlens")
        return layout


def _upstream():
    try:
        from xformers import ops
        from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
    except (ImportError, OSError) as exc:
        raise SpecError(
            "varlen CUTLASS requires a compatible optional xformers installation"
        ) from exc
    return ops, BlockDiagonalCausalMask


def _cutlass(query, key, value, layout, bias, *, scale=None):
    if not isinstance(layout, VarlenLayout):
        raise SpecError("varlen attention requires explicit validated sequence lengths")
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in (query, key, value)):
        raise SpecError("varlen Q/K/V must be [1, heads, total_tokens, head_dim]")
    if any(t.shape[0] != 1 or t.shape[2] != layout.total_tokens for t in (query, key, value)):
        raise SpecError("varlen Q/K/V token axes disagree with layout; KV cache is unsupported")
    if any(t.device != query.device or t.dtype != query.dtype for t in (key, value)):
        raise SpecError("varlen Q/K/V must share dtype/device")
    if query.device.type != "cuda" or query.dtype not in {torch.float16, torch.bfloat16}:
        raise SpecError("varlen CUTLASS requires CUDA fp16/bf16 Q/K/V")
    if key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise SpecError("varlen Q/K/V head dimensions are incompatible")
    if key.shape[1] == 0 or query.shape[1] % key.shape[1]:
        raise SpecError("varlen query heads must be a multiple of KV heads")
    # xFormers' 5D GQA path is forward-only. Explicit repetition keeps training
    # gradients to K/V; do not silently enter the inference-only 5D operation.
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    ops, _ = _upstream()
    try:
        return ops.memory_efficient_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_bias=bias,
            p=0.0,
            scale=scale,
            op=ops.MemoryEfficientAttentionCutlassOp,
        )
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        raise SpecError(f"requested varlen CUTLASS backend failed; no fallback: {exc}") from exc


def cutlass_varlen_attention(query, key, value, layout: VarlenLayout, *, scale=None):
    """BHTD inputs -> BTHD output, with block-isolated causal attention."""
    if not isinstance(layout, VarlenLayout):
        raise SpecError("varlen attention requires explicit validated sequence lengths")
    _, mask_type = _upstream()
    bias = mask_type.from_seqlens(layout.lengths, device=query.device)
    return _cutlass(query, key, value, layout, bias, scale=scale)


def _mask(
    *,
    batch_size,
    q_length,
    kv_length,
    attention_mask=None,
    q_offset=0,
    kv_offset=0,
    config=None,
    **kwargs,
):
    del kwargs
    if (
        batch_size != 1
        or q_length != kv_length
        or q_offset != 0
        or kv_offset != 0
        or attention_mask is not None
        or not getattr(config, "is_causal", True)
    ):
        raise SpecError("varlen mask only supports unpadded causal self-attention without KV cache")
    # A matching callback below enforces the mandatory per-forward layout.


def _attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    trainomni_varlen=None,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    if (
        attention_mask is not None
        or dropout != 0
        or kwargs.get("sliding_window") is not None
        or kwargs.get("softcap") is not None
        or kwargs.get("is_causal", True) is not True
        or not getattr(module.config, "is_causal", True)
    ):
        raise SpecError("varlen CUTLASS supports causal self-attention, no dropout/window/softcap")
    if not isinstance(trainomni_varlen, tuple) or len(trainomni_varlen) != 2:
        raise SpecError(
            "varlen model forward is missing validated layout; use padding_free_forward"
        )
    layout, bias = trainomni_varlen
    return _cutlass(query, key, value, layout, bias, scale=scaling), None


def padding_free_forward(model, *, inputs_embeds, layout: VarlenLayout):
    """Explicit Llama training/prefill boundary; call AFTER multimodal fusion.

    No generation, KV caching, sliding window, mRoPE, compile, or distributed
    claim. Vision retains its independently selected upstream attention backend.
    The setting is sticky: later direct forwards without layout fail closed.
    """
    from transformers import AttentionInterface, AttentionMaskInterface, LlamaForCausalLM

    if not isinstance(model, LlamaForCausalLM):
        raise SpecError("padding-free integration is currently verified for LlamaForCausalLM only")
    if not isinstance(layout, VarlenLayout):
        raise SpecError("padding-free forward requires VarlenLayout")
    if (
        inputs_embeds.ndim != 3
        or inputs_embeds.shape[:2] != (1, layout.total_tokens)
        or inputs_embeds.device.type != "cuda"
        or inputs_embeds.dtype not in {torch.float16, torch.bfloat16}
    ):
        raise SpecError("padding-free embeddings must be CUDA fp16/bf16 [1, total_tokens, hidden]")
    if (
        model.config.attention_dropout != 0
        or getattr(model.config, "sliding_window", None) is not None
        or not getattr(model.config, "is_causal", True)
    ):
        raise SpecError(
            "padding-free Llama requires causal attention without dropout/sliding window"
        )
    _, mask_type = _upstream()
    bias = mask_type.from_seqlens(layout.lengths, device=inputs_embeds.device)
    AttentionInterface.register(BACKEND, _attention)
    AttentionMaskInterface.register(BACKEND, _mask)
    if model.config._attn_implementation != BACKEND:
        model.set_attn_implementation(BACKEND)
    return model(
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        position_ids=layout.position_ids(inputs_embeds.device),
        trainomni_varlen=(layout, bias),
        use_cache=False,
        return_dict=True,
    )
