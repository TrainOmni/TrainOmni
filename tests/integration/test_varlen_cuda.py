"""Real upstream CUDA forward/backward; skips do not imply platform support."""

import pytest
import torch
import torch.nn.functional as F

from trainomni.core.errors import SpecError
from trainomni.runtime.kernels.attention.varlen import (
    VarlenLayout,
    cutlass_varlen_attention,
    padding_free_forward,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def require_xformers():
    pytest.importorskip("xformers.ops")


def oracle(q, k, v, lengths):
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    rows = []
    start = 0
    for n in lengths:
        rows.append(
            F.scaled_dot_product_attention(
                q[:, :, start : start + n].float(),
                k[:, :, start : start + n].float(),
                v[:, :, start : start + n].float(),
                is_causal=True,
            )
        )
        start += n
    return torch.cat(rows, dim=2).transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_heads", [2, 8])
def test_varlen_kernel_matches_separate_fp32_outputs_and_gradients(dtype, kv_heads):
    require_xformers()
    torch.manual_seed(1601)
    layout = VarlenLayout((1, 7, 13))
    q = torch.randn(1, 8, 21, 32, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, kv_heads, 21, 32, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = cutlass_varlen_attention(q, k, v, layout)
    expected = oracle(q, k, v, layout.lengths)
    torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * upstream).sum(), (q, k, v))
    reference_grads = torch.autograd.grad((expected * upstream.float()).sum(), (q, k, v))
    for grad, ref in zip(grads, reference_grads, strict=True):
        assert torch.isfinite(grad).all() and grad.norm() > 0
        error = (grad.float() - ref.float()).norm() / ref.float().norm()
        assert error.item() < 0.015


def test_varlen_isolation_causality_and_upstream_operator_evidence():
    require_xformers()
    torch.manual_seed(1602)
    layout = VarlenLayout((7, 13))
    q, k, v = [
        torch.randn(1, 4, 20, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        before = cutlass_varlen_attention(q, k, v, layout)
        before.float().sum().backward()
    names = {event.key for event in prof.key_averages()}
    assert "aten::_efficient_attention_forward" in names
    assert "aten::_efficient_attention_backward" in names
    assert not any("scaled_dot_product" in name for name in names)
    changed = v.detach().clone()
    changed[:, :, :7] += 100
    after = cutlass_varlen_attention(q.detach(), k.detach(), changed, layout)
    torch.testing.assert_close(before[:, 7:], after[:, 7:], rtol=0, atol=0)
    changed = v.detach().clone()
    changed[:, :, 6] += 100
    after = cutlass_varlen_attention(q.detach(), k.detach(), changed, layout)
    torch.testing.assert_close(before[:, :6], after[:, :6], rtol=0, atol=0)


def test_padding_free_llama_matches_separate_and_preserves_frozen_backbone_gradient():
    require_xformers()
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1603)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=97,
                attention_dropout=0.0,
            )
        )
        .cuda()
        .bfloat16()
        .eval()
        .requires_grad_(False)
    )
    layout = VarlenLayout((5, 9))
    embeddings = torch.randn(1, 14, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    model.set_attn_implementation("sdpa")
    reference = torch.cat(
        [
            model(inputs_embeds=part, use_cache=False).logits
            for part in embeddings.split(layout.lengths, dim=1)
        ],
        dim=1,
    )
    actual = padding_free_forward(model, inputs_embeds=embeddings, layout=layout).logits
    torch.testing.assert_close(actual, reference, atol=0.006, rtol=0.025)
    (ref_grad,) = torch.autograd.grad(reference.float().square().mean(), (embeddings,))
    (grad,) = torch.autograd.grad(actual.float().square().mean(), (embeddings,))
    assert (grad.float() - ref_grad.float()).norm() / ref_grad.float().norm() < 0.03
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(SpecError, match="missing validated layout"):
        model(inputs_embeds=embeddings, use_cache=False)
    with pytest.raises(SpecError, match="total_tokens"):
        padding_free_forward(model, inputs_embeds=embeddings, layout=VarlenLayout((13,)))


def test_varlen_no_cpu_or_padded_fallback():
    require_xformers()
    q = torch.randn(1, 4, 5, 32)
    with pytest.raises(SpecError, match="CUDA fp16/bf16"):
        cutlass_varlen_attention(q, q, q, VarlenLayout((5,)))
    q = q.cuda().bfloat16()
    with pytest.raises(SpecError, match="token axes"):
        cutlass_varlen_attention(q, q, q, VarlenLayout((4,)))


def test_varlen_backend_failure_does_not_fallback(monkeypatch):
    require_xformers()
    from xformers import ops

    def unavailable(*args, **kwargs):
        raise NotImplementedError("simulated incompatible CUDA build")

    monkeypatch.setattr(ops, "memory_efficient_attention", unavailable)
    q = torch.randn(1, 4, 5, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(SpecError, match="no fallback.*incompatible CUDA build"):
        cutlass_varlen_attention(q, q, q, VarlenLayout((5,)))
