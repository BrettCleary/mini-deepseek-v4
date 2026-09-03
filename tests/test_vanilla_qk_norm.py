"""QK-RMSNorm in the dense baseline.

CSA has applied RMSNorm to queries and compressed keys since Stage 2 (paper
§2.3.3, "avoids exploding attention logits"). Vanilla did not, which handed CSA
an arm-asymmetric stability advantage that grew with context: with
--grad-clip 1.0, vanilla's median gradient norm was 10.7 at 8K and 15.4 at 16K
(clipped on 91% / 94% of steps) while CSA sat near 0.6 and was essentially
never clipped. These tests pin the fix and its off-switch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402


def cfg(**kw):
    base = dict(attention="vanilla", vocab_size=64, d_model=64, n_layers=2,
                n_heads=4, block_size=32)
    base.update(kw)
    return model.ModelConfig(**base)


def test_qk_norm_is_on_by_default():
    attn = model.VanillaMultiHeadAttention(cfg())
    assert attn.qk_norm
    assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm")


def test_disabling_reproduces_the_original_state_dict():
    """Checkpoints written before this existed must still load, so the
    parameter set with the flag off has to be byte-identical to the old one."""
    off = model.MiniTransformer(cfg(vanilla_qk_norm=False))
    assert not any("q_norm" in k or "k_norm" in k for k in off.state_dict())
    on = model.MiniTransformer(cfg(vanilla_qk_norm=True))
    extra = set(on.state_dict()) - set(off.state_dict())
    assert extra == {
        f"blocks.{i}.attn.{p}_norm.weight" for i in range(2) for p in ("q", "k")
    }


def test_qk_norm_makes_query_scale_independent_of_input_scale():
    """The point of the fix: attention logits stop tracking activation
    magnitude, so they cannot grow without bound during training."""
    attn = model.VanillaMultiHeadAttention(cfg()).double()
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, dtype=torch.float64)

    def head_rms(scale):
        b, n, _ = x.shape
        qkv = attn.qkv(x * scale).view(b, n, 3, attn.n_heads, attn.head_dim)
        q, k, _ = qkv.unbind(dim=2)
        return (
            attn.q_norm(q).pow(2).mean(-1).sqrt().mean().item(),
            attn.k_norm(k).pow(2).mean(-1).sqrt().mean().item(),
        )

    small_q, small_k = head_rms(1.0)
    large_q, large_k = head_rms(1000.0)
    # Not exactly equal: RMSNorm's eps=1e-5 sits inside the sqrt, so it shifts
    # the result slightly at small input scale and vanishes at large scale.
    # That eps floor is the size of the residual difference here.
    assert abs(small_q - large_q) < 1e-4, (small_q, large_q)
    assert abs(small_k - large_k) < 1e-4, (small_k, large_k)
    # A 1000x change in input scale must not move the query scale by anything
    # like 1000x -- that is the growth the fix removes.
    assert abs(small_q - 1.0) < 1e-4


def test_qk_norm_bounds_logit_growth_when_weights_blow_up():
    """Simulate the failure mode: scale the qkv projection up hard and check
    the normalised path keeps attention finite and well-behaved."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, dtype=torch.float64)
    outs = {}
    for flag in (False, True):
        torch.manual_seed(7)
        attn = model.VanillaMultiHeadAttention(cfg(vanilla_qk_norm=flag)).double()
        with torch.no_grad():
            attn.qkv.weight.mul_(50.0)
        outs[flag] = attn(x).abs().max().item()
    assert torch.isfinite(torch.tensor(outs[True]))
    # Unnormalised logits scale with the square of the blown-up projection.
    assert outs[True] < outs[False]


def test_model_runs_and_backprops_with_qk_norm():
    m = model.MiniTransformer(cfg())
    idx = torch.randint(0, 64, (2, 32))
    logits, loss, _ = m(idx, idx)
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert m.blocks[0].attn.q_norm.weight.grad is not None
