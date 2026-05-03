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
    out = attn(x)

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
        out_a = attn(x)

    # Perturb a "future" token: t' = block_size - 1.
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    with torch.no_grad():
        out_b = attn(x2)

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
# Runner (no pytest required)
# ---------------------------------------------------------------------------


def _run_all() -> None:
    tests = [
        test_compression_shape,
        test_compression_handcomputed_m2,
        test_compression_block_zero_boundary_no_cb_contribution,
        test_compression_softmax_columns_sum_to_one,
        test_compression_overlap_invariant,
        test_attention_first_m_output_is_zero,
        test_attention_causal_mask_shape,
        test_attention_no_future_leakage,
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
