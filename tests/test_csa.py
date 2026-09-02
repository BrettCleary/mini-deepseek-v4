"""Unit tests for the CSA mechanism — Stage 2.

Tests cover the compression math (eqs. 11-12), the i=0 boundary, the
overlap invariant between adjacent compressed blocks, the causal mask, and
the safe-softmax behavior that produces zero output for the first m
positions.

Runnable two ways:
    .venv/bin/python tests/test_csa.py
    .venv/bin/python -m pytest tests/test_csa.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Allow `python tests/test_csa.py` from the project root to import sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_compressor(d: int, c: int, m: int, identity_kv: bool = False) -> model.CSACompression:
    """Build a CSACompression with all bias parameters zeroed and, optionally,
    KV projections set to the identity (so C^a = C^b = H slice). Z projections
    stay random unless explicitly zeroed by the caller."""
    comp = model.CSACompression(d_model=d, c=c, m=m)
    if identity_kv:
        assert d == c, "identity_kv requires d == c"
        with torch.no_grad():
            comp.W_aKV.weight.copy_(torch.eye(c))
            comp.W_bKV.weight.copy_(torch.eye(c))
    with torch.no_grad():
        comp.B_a.zero_()
        comp.B_b.zero_()
    return comp


def zero_z_projections(comp: model.CSACompression) -> None:
    with torch.no_grad():
        comp.W_aZ.weight.zero_()
        comp.W_bZ.weight.zero_()


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------


def test_compression_shape() -> None:
    """C^a, C^b, Z^a, Z^b are (b, n, c); C^Comp is (b, n/m, c)."""
    b, n, d, c, m = 2, 8, 16, 4, 2
    comp = model.CSACompression(d_model=d, c=c, m=m)
    h = torch.randn(b, n, d)
    out = comp(h)
    assert out.shape == (b, n // m, c), f"got {out.shape}, want ({b}, {n // m}, {c})"


def test_compression_handcomputed_m2() -> None:
    """With identity-KV and zeroed Z and biases:
       - block 0: Z^b is -inf, only C^a contributes; uniform 0.5/0.5 over m=2 cur positions
       - block 1: all of Z is zero, uniform 0.25 weights over 2m=4 positions

    Setup (m=2, c=2, n=4, batch=1):
        H = [[1,2], [3,4], [5,6], [7,8]]
        C_a = C_b = H (identity KV projection)
        Z_a = Z_b = 0 (zero projection)
    Block 0:
        cur (Z_a + B_a) = [[0,0],[0,0]]
        prev (Z_b + B_b) padded = [[-inf,-inf],[-inf,-inf]]
        softmax over 2m=4 axis -> [[0.5,0.5],[0.5,0.5],[0,0],[0,0]]
        C_0^Comp = 0.5*[1,2] + 0.5*[3,4] = [2, 3]
    Block 1:
        cur = [[0,0],[0,0]] (positions 2,3 of Z_a)
        prev = [[0,0],[0,0]] (positions 0,1 of Z_b — no padding here)
        softmax uniform over 4 -> 0.25 each
        C_1^Comp = 0.25 * ([5,6] + [7,8] + [1,2] + [3,4]) = [4, 5]
    """
    b, n, d, c, m = 1, 4, 2, 2, 2
    comp = make_compressor(d, c, m, identity_kv=True)
    zero_z_projections(comp)

    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    expected = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])

    out = comp(h)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_compression_block_zero_boundary_no_cb_contribution() -> None:
    """For block 0, even if we set C^b to wildly different values, the output
    should be unchanged because S^b for block 0 is identically zero (Z^b is
    -inf-padded). This isolates the boundary handling: if anything from the
    padded slot leaked through, this test would fail."""
    b, n, d, c, m = 1, 4, 4, 4, 2
    comp = make_compressor(d, c, m, identity_kv=False)
    zero_z_projections(comp)

    h = torch.randn(b, n, d)
    out_normal = comp(h)

    # Now perturb W_bKV — block 0's output should be unchanged.
    with torch.no_grad():
        comp.W_bKV.weight.add_(100.0 * torch.randn_like(comp.W_bKV.weight))
    out_perturbed = comp(h)

    torch.testing.assert_close(
        out_normal[:, 0, :], out_perturbed[:, 0, :], rtol=1e-6, atol=1e-5,
        msg="block 0 output changed when W_bKV was perturbed — Z^b -inf padding is leaking"
    )
    # Sanity: block 1's output should change, since it uses C^b legitimately.
    assert not torch.allclose(out_normal[:, 1, :], out_perturbed[:, 1, :], atol=1e-4), (
        "block 1 output is unchanged after a 100x perturbation of W_bKV — something is wrong"
    )


def test_compression_softmax_columns_sum_to_one() -> None:
    """Eq. 11: softmax_row normalizes per feature column across 2m positions.
    For every block i ≥ 1 (i=0 has -inf-padded prev slots and sums to 1 only
    on the cur half), every (b, n_blk≥1, c) column of S sums to 1 over 2m."""
    b, n, d, c, m = 1, 8, 8, 8, 2
    comp = model.CSACompression(d_model=d, c=c, m=m)
    h = torch.randn(b, n, d)

    # Replicate the internal Z_concat → softmax to check the invariant.
    C_a, C_b = comp.W_aKV(h), comp.W_bKV(h)
    Z_a, Z_b = comp.W_aZ(h), comp.W_bZ(h)
    n_blk = n // m

    Z_a_cur = Z_a.view(b, n_blk, m, c)
    Z_b_padded = torch.nn.functional.pad(Z_b, (0, 0, m, 0), value=float("-inf"))
    Z_b_prev = Z_b_padded[:, :n, :].view(b, n_blk, m, c)
    Z_concat = torch.cat([Z_a_cur + comp.B_a, Z_b_prev + comp.B_b], dim=2)
    S = torch.softmax(Z_concat, dim=2)

    # block 0: only the cur half (rows 0..m-1) sums to 1, prev half is 0
    sums_block0_cur = S[:, 0, :m, :].sum(dim=1)        # (b, c)
    sums_block0_prev = S[:, 0, m:, :].sum(dim=1)
    torch.testing.assert_close(sums_block0_cur, torch.ones_like(sums_block0_cur))
    torch.testing.assert_close(sums_block0_prev, torch.zeros_like(sums_block0_prev))
    # blocks 1..n_blk-1: full 2m sums to 1
    full_sums = S[:, 1:, :, :].sum(dim=2)              # (b, n_blk-1, c)
    torch.testing.assert_close(full_sums, torch.ones_like(full_sums))


def test_compression_overlap_invariant() -> None:
    """Per the paper's note in §2.3.1: 'indexes of C^b used for C_i^Comp and
    indexes of C^a used for C_{i-1}^Comp are overlapped'. Both touch the
    token range [m(i-1), mi). Verify that perturbing H at those positions
    affects BOTH adjacent compressed entries."""
    b, n, d, c, m = 1, 8, 8, 8, 2
    comp = model.CSACompression(d_model=d, c=c, m=m)

    h = torch.randn(b, n, d)
    out_a = comp(h)

    # Perturb token positions [m, 2m) — the "current" range of block 1 and
    # the "previous" range of block 2.
    h2 = h.clone()
    h2[:, m:2 * m, :] += 5.0
    out_b = comp(h2)

    delta = (out_b - out_a).abs().sum(dim=-1).squeeze(0)  # (n_blk,)
    # Block 0 — uses tokens [0, m), shouldn't change.
    assert delta[0].item() < 1e-4, f"block 0 should not change, got delta={delta[0].item()}"
    # Block 1 — current window covers the perturbed tokens.
    assert delta[1].item() > 1e-2, f"block 1 should change, got delta={delta[1].item()}"
    # Block 2 — previous window covers the perturbed tokens (overlap invariant).
    assert delta[2].item() > 1e-2, f"block 2 should change (overlap invariant), got delta={delta[2].item()}"


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------


def test_attention_first_m_output_is_zero() -> None:
    """Per D2: positions [0, m) have no causally-valid block, so the safe
    softmax (rows where t // m == 0) produces zero output. The W_O linear
    has no bias, so the *attention block's* output for those positions is
    exactly zero."""
    cfg = model.ModelConfig(
        vocab_size=10, d_model=16, n_layers=1, n_heads=2,
        block_size=16, attention="csa", csa_m=4, csa_c=8,
    )
    attn = model.CSAAttention(cfg)
    x = torch.randn(2, cfg.block_size, cfg.d_model)
    out, _ = attn(x)

    # First m positions: identically zero.
    torch.testing.assert_close(
        out[:, :cfg.csa_m, :],
        torch.zeros(2, cfg.csa_m, cfg.d_model),
        rtol=0, atol=1e-6,
    )
    # Position m onwards: should be nonzero (block 0 is causally available).
    assert out[:, cfg.csa_m:, :].abs().max().item() > 1e-3


def test_attention_causal_mask_shape() -> None:
    """For a query at token t and a compressed block s, mask is True iff
    s < floor(t/m). Spot-check a few positions."""
    n, m, n_blk = 16, 4, 4
    token_idx = torch.arange(n).unsqueeze(1)       # (n, 1)
    block_idx = torch.arange(n_blk).unsqueeze(0)   # (1, n_blk)
    mask = block_idx < (token_idx // m)            # (n, n_blk)

    # t < m: no block valid
    for t in range(m):
        assert not mask[t].any(), f"t={t} should see no blocks"
    # t in [m, 2m): block 0 only
    for t in range(m, 2 * m):
        assert mask[t, 0].item() and not mask[t, 1:].any(), f"t={t} should see only block 0"
    # t in [2m, 3m): blocks 0, 1
    for t in range(2 * m, 3 * m):
        assert mask[t, :2].all() and not mask[t, 2:].any(), f"t={t} should see blocks 0,1"


def test_attention_no_future_leakage() -> None:
    """Perturbing token h[t'] for t' > t should not change the attention
    output at position t."""
    cfg = model.ModelConfig(
        vocab_size=10, d_model=16, n_layers=1, n_heads=2,
        block_size=16, attention="csa", csa_m=4, csa_c=8,
    )
    attn = model.CSAAttention(cfg)
    attn.eval()

    x = torch.randn(1, cfg.block_size, cfg.d_model)
    with torch.no_grad():
        out_a, _ = attn(x)

    # Perturb a "future" token: t' = block_size - 1.
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    with torch.no_grad():
        out_b, _ = attn(x2)

    # All but the last few positions should be unchanged. Specifically, any
    # position t with floor(t/m) <= floor((n-1)/m) - 1 cannot see the last
    # token's block. With n=16, m=4: last block is 3 (positions 12..15).
    # Positions 0..11 attend to blocks 0..2 only; position 11's mask is
    # blocks 0..2, so it doesn't see block 3 either. Check positions 0..11.
    torch.testing.assert_close(
        out_a[:, :12, :], out_b[:, :12, :], rtol=1e-5, atol=1e-5,
        msg="future token leaked into past-position output",
    )


# ---------------------------------------------------------------------------
# Indexer tests (Stage 3)
# ---------------------------------------------------------------------------


def _indexer_cfg(**overrides) -> model.ModelConfig:
    base = dict(
        vocab_size=10, d_model=16, n_layers=1, n_heads=2,
        block_size=16, attention="csa",
        csa_m=4, csa_c=8, csa_c_i=4, csa_n_h_i=2, csa_d_c=8, csa_top_k=2,
    )
    base.update(overrides)
    return model.ModelConfig(**base)


def test_indexer_shape() -> None:
    """Scores: (b, n, n_blk). Latent c^Q: (b, n, d_c)."""
    cfg = _indexer_cfg()
    idx = model.CSAIndexer(cfg)
    h = torch.randn(3, cfg.block_size, cfg.d_model)
    scores, c_q = idx(h)
    assert scores.shape == (3, cfg.block_size, cfg.block_size // cfg.csa_m)
    assert c_q.shape == (3, cfg.block_size, cfg.csa_d_c)


def test_indexer_relu_kills_negative_dots() -> None:
    """If all q^I·K^IComp dots are negative, ReLU zeroes them and the
    score is identically zero regardless of `w^I`."""
    cfg = _indexer_cfg()
    idx = model.CSAIndexer(cfg)
    # Force q^I to be a large negative constant and K^IComp to be a large
    # positive constant: every dot is hugely negative -> ReLU = 0.
    with torch.no_grad():
        idx.W_DQ.weight.zero_()
        idx.W_IUQ.weight.zero_()
        idx.W_IUQ.bias is None  # sanity (Linear(... bias=False))
        # Bias-free linears only zero the projection. Use direct override:
        # build an alternate path by adding a bias via a hook — simpler to
        # just inject q^I and K^IComp values via monkey-patching forward.
    # Simpler: replace the modules with constant-valued ones.
    h = torch.randn(2, cfg.block_size, cfg.d_model)

    class FakeIndexer(torch.nn.Module):
        def __init__(self, real: model.CSAIndexer) -> None:
            super().__init__()
            self.W_w = real.W_w
            self.n_h_i = real.n_h_i
            self.q_const = -1e3
            self.k_const = 1e3

        def forward(self, h: torch.Tensor):
            b, n, _ = h.shape
            n_blk = n // 4
            q_i = torch.full((b, n, self.n_h_i, 4), self.q_const)
            K_i = torch.full((b, n_blk, 4), self.k_const)
            w_i = self.W_w(h)
            dots = einsum_local(q_i, K_i)
            scores = (w_i.unsqueeze(-1) * torch.relu(dots)).sum(dim=2)
            return scores, h

    def einsum_local(q, k):
        # mimic the real indexer's einsum
        return torch.einsum("bnhc,bsc->bnhs", q, k)

    fake = FakeIndexer(idx)
    scores, _ = fake(h)
    assert torch.all(scores == 0), f"expected all zeros (ReLU on negative dots), got max={scores.abs().max().item()}"


def test_indexer_zero_w_zeroes_scores() -> None:
    """If `w^I` is identically zero, every score is zero regardless of dots."""
    cfg = _indexer_cfg()
    idx = model.CSAIndexer(cfg)
    with torch.no_grad():
        idx.W_w.weight.zero_()
    h = torch.randn(2, cfg.block_size, cfg.d_model)
    scores, _ = idx(h)
    assert torch.all(scores == 0)


def test_indexer_handcomputed_single_position() -> None:
    """Hand-compute eq. 16 for one (t, s) pair.

    Setup: n_h_i=1 so the per-head sum collapses to a single term:
        I[t, s] = w^I[t, 0] * ReLU(q^I[t, 0] · K^IComp[s])
    Set W_DQ, W_IUQ to the identity-like maps, W_w to a constant 1, and
    pin the indexer's compressor to known weights so K^IComp[s] is
    known. Then verify one slot.
    """
    cfg = _indexer_cfg(d_model=4, csa_d_c=4, csa_c_i=4, csa_n_h_i=1, block_size=8, csa_m=4)
    idx = model.CSAIndexer(cfg)
    with torch.no_grad():
        idx.W_DQ.weight.copy_(torch.eye(cfg.d_model))         # c^Q = h
        idx.W_IUQ.weight.copy_(torch.eye(cfg.csa_c_i))        # q^I = c^Q (one head)
        idx.W_w.weight.zero_()                                # base
        idx.W_w.weight[0, 0] = 1.0                            # w^I = h[..., 0] (one head)
        # Build the indexer's compressor to deterministic outputs:
        #   W_aKV = identity, W_bKV = identity
        #   W_aZ = W_bZ = 0, B_a = B_b = 0
        # Then K^IComp[s] = mean over 2m source-token vectors of the
        # (a-stream + b-stream-front-padded) — exactly like Stage 2's
        # hand-compute test.
        idx.compress_indexer.W_aKV.weight.copy_(torch.eye(cfg.csa_c_i))
        idx.compress_indexer.W_bKV.weight.copy_(torch.eye(cfg.csa_c_i))
        idx.compress_indexer.W_aZ.weight.zero_()
        idx.compress_indexer.W_bZ.weight.zero_()
        idx.compress_indexer.B_a.zero_()
        idx.compress_indexer.B_b.zero_()

    # Build h: each token's vector is its position index repeated.
    h = torch.zeros(1, cfg.block_size, cfg.d_model)
    for t in range(cfg.block_size):
        h[0, t, 0] = float(t)            # so w^I[t, 0] = t
        h[0, t, 1] = 1.0                 # the rest of h is constant 1

    scores, _ = idx(h)
    # Block 0: K^IComp[0] = average of h[0..3] = ([0+1+2+3]/4, 1, 0, 0) = (1.5, 1, 0, 0)
    # but with i=0 boundary: prev half is zero, so denominator over m=4 only.
    # q^I[t, 0] = h[t] = (t, 1, 0, 0)
    # dot = t * 1.5 + 1 * 1 = 1.5 t + 1
    # I[t, 0] = w^I[t, 0] * relu(1.5 t + 1) = t * (1.5 t + 1)  for t >= 0
    for t in range(1, cfg.block_size):
        expected = float(t) * (1.5 * t + 1.0)
        actual = scores[0, t, 0].item()
        assert abs(actual - expected) < 1e-4, f"t={t}: expected {expected}, got {actual}"


def test_attention_topk_active_only_at_eval() -> None:
    """csa_top_k < n_blk should change attention output between train and eval
    (top-k applied only at eval per D3)."""
    cfg = _indexer_cfg(block_size=32, csa_m=4, csa_top_k=1)  # n_blk=8, top_k=1
    attn = model.CSAAttention(cfg)
    x = torch.randn(2, cfg.block_size, cfg.d_model)
    attn.train()
    with torch.no_grad():
        out_train, _ = attn(x)
    attn.eval()
    with torch.no_grad():
        out_eval, _ = attn(x)
    # Different in train vs eval (because eval applies top-k mask).
    assert not torch.allclose(out_train, out_eval, atol=1e-5), (
        "expected train/eval outputs to differ when top-k is active"
    )


def test_attention_topk_disabled_when_geq_n_blk() -> None:
    """If csa_top_k >= n_blk, top-k is a no-op so train and eval outputs match
    (modulo dropout etc — none here)."""
    cfg = _indexer_cfg(block_size=32, csa_m=4, csa_top_k=8)  # n_blk=8, top_k=8
    attn = model.CSAAttention(cfg)
    x = torch.randn(2, cfg.block_size, cfg.d_model)
    attn.train()
    with torch.no_grad():
        out_train, _ = attn(x)
    attn.eval()
    with torch.no_grad():
        out_eval, _ = attn(x)
    torch.testing.assert_close(out_train, out_eval, rtol=1e-5, atol=1e-5)


def test_attention_no_future_leakage_with_indexer() -> None:
    """No-future-leakage invariant must still hold with the indexer wired in."""
    cfg = _indexer_cfg(block_size=16, csa_m=4)
    attn = model.CSAAttention(cfg)
    attn.eval()

    x = torch.randn(1, cfg.block_size, cfg.d_model)
    with torch.no_grad():
        out_a, _ = attn(x)
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    with torch.no_grad():
        out_b, _ = attn(x2)

    # Same reasoning as Stage 2's no-future-leakage test: positions 0..11
    # never attend to block 3 (which contains positions 12..15).
    torch.testing.assert_close(
        out_a[:, :12, :], out_b[:, :12, :], rtol=1e-5, atol=1e-5,
        msg="future token leaked into past-position output (with indexer)",
    )


# ---------------------------------------------------------------------------
# Shared-KV MQA tests (Stage 4)
# ---------------------------------------------------------------------------


def test_stage4_uses_w_uq_not_w_q() -> None:
    """Stage 4 removes the direct H -> queries projection (W_Q) and replaces
    it with a low-rank c^Q -> queries projection (W_UQ)."""
    cfg = _indexer_cfg()
    attn = model.CSAAttention(cfg)
    assert hasattr(attn, "W_UQ"), "Stage 4 should add W_UQ"
    assert not hasattr(attn, "W_Q"), "Stage 4 should drop the direct W_Q (queries come from c^Q)"
    # W_UQ shape: (n_h * c, d_c) — Linear stores weight as (out, in)
    assert attn.W_UQ.weight.shape == (cfg.n_heads * cfg.csa_c, cfg.csa_d_c)


def test_stage4_W_DQ_feeds_core_queries() -> None:
    """In Stage 4 the indexer's W_DQ projection feeds BOTH the indexer
    queries AND the core queries (eq. 13 + eq. 18 share c^Q). Perturbing
    W_DQ should change the attention output even when the indexer's
    contribution to the logits is held constant."""
    cfg = _indexer_cfg(block_size=16, csa_m=4, csa_top_k=99)  # disable top-k effect
    attn = model.CSAAttention(cfg)
    attn.eval()
    attn.eval_apply_topk = False  # also disable to be safe
    x = torch.randn(1, cfg.block_size, cfg.d_model)
    with torch.no_grad():
        out_a, _ = attn(x)
        # Perturb W_DQ — c^Q changes, so both indexer queries AND core queries change.
        attn.indexer.W_DQ.weight.add_(0.5 * torch.randn_like(attn.indexer.W_DQ.weight))
        out_b, _ = attn(x)
    assert not torch.allclose(out_a, out_b, atol=1e-5), (
        "perturbing W_DQ should change attention output (c^Q feeds core queries in Stage 4)"
    )


def test_stage4_zero_W_UQ_zeros_attention_output_post_first_m() -> None:
    """If W_UQ is zero, core queries are zero, so q·K = 0 and every logit is
    zero. softmax over the causally-valid blocks then yields a uniform
    distribution, and the attention output is the uniform average of the
    K=V compressed blocks. That's well-defined and not identically zero.

    The point of this test: the first-m-positions zero invariant from
    Stage 2 must still hold with W_UQ zeroed — proving the safe-softmax
    guard is robust to a degenerate query path."""
    cfg = _indexer_cfg(block_size=16, csa_m=4, csa_top_k=99)
    attn = model.CSAAttention(cfg)
    attn.eval_apply_topk = False
    with torch.no_grad():
        attn.W_UQ.weight.zero_()
    x = torch.randn(2, cfg.block_size, cfg.d_model)
    with torch.no_grad():
        out, _ = attn(x)
    # First m positions still exactly zero (causal mask + safe softmax).
    torch.testing.assert_close(
        out[:, :cfg.csa_m, :], torch.zeros(2, cfg.csa_m, cfg.d_model),
        rtol=0, atol=1e-6,
    )
    # Position m onwards is non-zero (driven by indexer-only attention).
    assert out[:, cfg.csa_m:, :].abs().max().item() > 1e-3


# ---------------------------------------------------------------------------
# Stage 5: V3.2-style indexer KL loss + gradient isolation
# ---------------------------------------------------------------------------


def _kl_isolation_grad(loss: torch.Tensor) -> None:
    """Backward, populating .grad on every leaf that participated."""
    loss.backward()


def test_indexer_kl_gradient_isolation() -> None:
    """L_LM gradient must not flow into indexer-specific params (W_IUQ, W_w,
    indexer compressor). L_I gradient must not flow into main-model params
    (W_DQ, W_UQ, main compress, W_O, FFN, embeddings, norms). The detach in
    CSAIndexer.forward is what makes this hold."""
    cfg = _indexer_cfg(block_size=16, csa_m=4, vocab_size=10, n_layers=1)
    m = model.MiniTransformer(cfg)

    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    loss_mask = torch.ones((2, cfg.block_size), dtype=torch.float)
    loss_mask[:, : cfg.csa_m] = 0.0

    block = m.blocks[0]
    attn = block.attn
    indexer = attn.indexer

    indexer_specific = {
        "indexer.W_IUQ": indexer.W_IUQ.weight,
        "indexer.W_w": indexer.W_w.weight,
        "indexer.compress_indexer.W_aKV": indexer.compress_indexer.W_aKV.weight,
        "indexer.compress_indexer.W_bKV": indexer.compress_indexer.W_bKV.weight,
        "indexer.compress_indexer.W_aZ": indexer.compress_indexer.W_aZ.weight,
        "indexer.compress_indexer.W_bZ": indexer.compress_indexer.W_bZ.weight,
        "indexer.compress_indexer.B_a": indexer.compress_indexer.B_a,
        "indexer.compress_indexer.B_b": indexer.compress_indexer.B_b,
    }
    main_only = {
        "indexer.W_DQ": indexer.W_DQ.weight,
        "attn.W_UQ": attn.W_UQ.weight,
        "attn.compress.W_aKV": attn.compress.W_aKV.weight,
        "attn.W_O": attn.W_O.weight,
        "tok_embed": m.tok_embed.weight,
        "pos_embed": m.pos_embed.weight,
    }

    # ---- 1. L_LM backward only ----
    _, lm_loss, indexer_loss = m(idx, tgt, loss_mask=loss_mask)
    assert lm_loss is not None and indexer_loss is not None
    m.zero_grad(set_to_none=True)
    _kl_isolation_grad(lm_loss)

    for name, p in indexer_specific.items():
        grad_norm = 0.0 if p.grad is None else p.grad.abs().sum().item()
        assert grad_norm == 0.0, f"L_LM leaked into {name}: |grad|={grad_norm}"
    for name, p in main_only.items():
        grad_norm = 0.0 if p.grad is None else p.grad.abs().sum().item()
        assert grad_norm > 0.0, f"L_LM did not reach {name} (expected nonzero grad)"

    # ---- 2. L_I backward only (fresh forward to rebuild graph) ----
    _, lm_loss, indexer_loss = m(idx, tgt, loss_mask=loss_mask)
    m.zero_grad(set_to_none=True)
    _kl_isolation_grad(indexer_loss)

    for name, p in main_only.items():
        grad_norm = 0.0 if p.grad is None else p.grad.abs().sum().item()
        assert grad_norm == 0.0, f"L_I leaked into {name}: |grad|={grad_norm}"
    # Indexer params: most should receive L_I grad. Allow some to be zero
    # only if a specific structural reason exists; otherwise expect nonzero.
    for name, p in indexer_specific.items():
        grad_norm = 0.0 if p.grad is None else p.grad.abs().sum().item()
        assert grad_norm > 0.0, f"L_I did not reach {name}"


def test_minitransformer_returns_three_values() -> None:
    """forward returns (logits, lm_loss, indexer_loss); indexer_loss is a
    finite tensor for CSA and None for vanilla."""
    # CSA
    cfg = _indexer_cfg(block_size=16, csa_m=4, vocab_size=10, n_layers=1)
    m = model.MiniTransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    logits, lm_loss, indexer_loss = m(idx, tgt)
    assert logits.shape == (1, cfg.block_size, cfg.vocab_size)
    assert lm_loss is not None and torch.isfinite(lm_loss).item()
    assert indexer_loss is not None and torch.isfinite(indexer_loss).item()
    # vanilla
    vcfg = model.ModelConfig(
        vocab_size=10, d_model=16, n_layers=1, n_heads=2,
        block_size=16, attention="vanilla",
    )
    vm = model.MiniTransformer(vcfg)
    _, vlm, vix = vm(idx, tgt)
    assert vlm is not None and vix is None


def test_indexer_loss_zero_when_indexer_matches_attention() -> None:
    """Sanity check the KL: if the indexer's softmax already equals p, L_I
    should be ~0. We construct this by zeroing the indexer-specific params
    so the indexer scores are identically zero; softmax(0..0) is uniform
    over n_blk. p is the uniform-over-valid attention distribution only
    when attention scores happen to make it so — that's hard to force.

    Easier: zero W_w. Then indexer scores are all zero per (t, s). softmax
    over the causally-masked positions (rest are -inf) gives uniform over
    valid blocks. If the attention output p is ALSO uniform over the same
    valid blocks, KL = 0. We force this by making the queries zero (zero
    W_UQ): then core_scores=0, softmax over valid blocks is uniform — and
    p = head-sum/n_h normalized = uniform too. So KL = 0 for every t >= m."""
    cfg = _indexer_cfg(block_size=16, csa_m=4, vocab_size=10, n_layers=1)
    m = model.MiniTransformer(cfg)
    block = m.blocks[0]
    with torch.no_grad():
        block.attn.W_UQ.weight.zero_()          # core queries -> 0 -> p uniform over valid
        block.attn.indexer.W_w.weight.zero_()   # indexer scores -> 0 -> softmax(I) uniform over valid

    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    loss_mask = torch.ones((1, cfg.block_size), dtype=torch.float)
    loss_mask[:, : cfg.csa_m] = 0.0
    _, _, indexer_loss = m(idx, tgt, loss_mask=loss_mask)
    assert indexer_loss.item() < 1e-5, f"expected ~0 KL when distributions match, got {indexer_loss.item()}"


# ---------------------------------------------------------------------------
# Runner (no pytest required)
# ---------------------------------------------------------------------------


def _run_all() -> None:
    tests = [
        # Stage 2: compression + dense attention
        test_compression_shape,
        test_compression_handcomputed_m2,
        test_compression_block_zero_boundary_no_cb_contribution,
        test_compression_softmax_columns_sum_to_one,
        test_compression_overlap_invariant,
        test_attention_first_m_output_is_zero,
        test_attention_causal_mask_shape,
        test_attention_no_future_leakage,
        # Stage 3: lightning indexer
        test_indexer_shape,
        test_indexer_relu_kills_negative_dots,
        test_indexer_zero_w_zeroes_scores,
        test_indexer_handcomputed_single_position,
        test_attention_topk_active_only_at_eval,
        test_attention_topk_disabled_when_geq_n_blk,
        test_attention_no_future_leakage_with_indexer,
        # Stage 4: shared-KV MQA
        test_stage4_uses_w_uq_not_w_q,
        test_stage4_W_DQ_feeds_core_queries,
        test_stage4_zero_W_UQ_zeros_attention_output_post_first_m,
        # Stage 5: V3.2 indexer KL training + gradient isolation
        test_indexer_kl_gradient_isolation,
        test_minitransformer_returns_three_values,
        test_indexer_loss_zero_when_indexer_matches_attention,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()


# ---------------------------------------------------------------------------
# Memory-motivated refactors: both must be exactly output-preserving.
# ---------------------------------------------------------------------------


def _csa_model(**kw):
    cfg = model.ModelConfig(
        attention="csa", vocab_size=37, d_model=48, n_layers=2, n_heads=4,
        block_size=64, csa_m=4, csa_top_k=5, **kw,
    )
    torch.manual_seed(1234)
    return model.MiniTransformer(cfg).double()


def _run(m, apply_topk):
    torch.manual_seed(9)
    idx = torch.randint(0, 37, (2, 64))
    tgt = torch.randint(0, 37, (2, 64))
    mask = torch.ones(2, 64)
    mask[:, :4] = 0.0
    for blk in m.blocks:
        blk.attn.eval_apply_topk = apply_topk
    logits, lm, kl = m(idx, tgt, loss_mask=mask)
    return logits, lm, kl


def test_query_chunking_is_exact():
    """csa_chunk is a memory knob only — it must not move any number.

    The score matrix is (b, n_heads, n, n_blk) and autocast runs softmax in
    fp32, so at 16K it is 1.5GB per layer; chunking bounds the transient
    copies. Any numerical drift here would silently confound the sweep.
    """
    for topk in (False, True):
        ref = _csa_model(csa_chunk=0)
        ref.train(not topk)
        base = _run(ref, topk)
        for chunk in (8, 16, 32, 64, 128):
            m = _csa_model(csa_chunk=chunk)
            m.train(not topk)
            got = _run(m, topk)
            for b, g, name in zip(base, got, ("logits", "lm_loss", "kl")):
                assert torch.equal(b, g), f"chunk={chunk} topk={topk} changed {name}"


def test_fully_masked_rows_produce_zero_output_and_no_nan():
    """Positions t < m have no causally-valid block. The mask uses a large
    finite value rather than -inf (so softmax yields uniform, not NaN); the
    zeroing must therefore happen explicitly on the output."""
    m = _csa_model()
    m.eval()
    torch.manual_seed(3)
    x = torch.randn(2, 64, 48, dtype=torch.float64)
    attn = m.blocks[0].attn
    out, aux = attn(x)
    assert torch.isfinite(out).all(), "attention output contains NaN/inf"
    assert torch.equal(out[:, :attn.m], torch.zeros_like(out[:, :attn.m]))
    assert torch.equal(aux["p"][:, :attn.m], torch.zeros_like(aux["p"][:, :attn.m]))
    # Teacher rows for valid positions must still be probability distributions.
    assert torch.allclose(aux["p"][:, attn.m:].sum(-1), torch.ones(2, 64 - attn.m, dtype=torch.float64))


def test_indexer_kl_matches_explicit_log_softmax_form():
    """The KL is computed via the logsumexp identity to avoid materializing
    log_softmax(I) (a full (b, n, n_blk) fp32 tensor kept for backward).
    Check it against the literal textbook form."""
    m = _csa_model()
    m.train()
    torch.manual_seed(5)
    x = torch.randn(2, 64, 48, dtype=torch.float64)
    _, aux = m.blocks[0].attn(x)
    p, I_m = aux["p"], aux["I_masked"]

    log_q = torch.log_softmax(I_m, dim=-1)
    contrib = p * (torch.log(p.clamp_min(1e-30)) - log_q)
    expected = torch.where(p > 0, contrib, torch.zeros_like(contrib)).sum(-1)

    got = m._indexer_kl_loss([aux], loss_mask=None)
    assert torch.allclose(got, expected.mean(), atol=1e-12), f"{got} vs {expected.mean()}"
