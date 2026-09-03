"""Decoder-only transformer for the mini-deepseek-v4 study.

The attention block is the single component that varies between the vanilla
baseline (Stage 1) and CSA (Stages 2-4); everything else — embeddings,
RMSNorm, SwiGLU MLP, output head — is shared so the comparison isolates the
attention mechanism.

Per design decision D4 (notes.md): no RoPE — learned absolute position
embeddings, identical for both architectures.

Stage 2 implements CSA's compression pipeline (paper eqs. 9-12) and dense
attention over all causally-valid compressed blocks. The lightning indexer
(Stage 3, eqs. 13-17) and the latent-query factorization (Stage 4, eqs. 18-19)
arrive in subsequent commits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    block_size: int = 1024
    attention: str = "vanilla"  # "vanilla" | "csa"
    d_ff: int | None = None     # SwiGLU inner dim. Defaults to ~8/3 * d_model.

    # CSA-specific (used only when attention == "csa")
    csa_m: int = 4                # compression factor: m KV tokens -> 1 compressed entry
    csa_c: int | None = None      # compressed/per-head dim. Defaults to head_dim.
    csa_c_i: int | None = None    # indexer head dim. Defaults to csa_c // 2.
    csa_n_h_i: int | None = None  # indexer query heads. Defaults to max(2, n_heads // 2).
    csa_d_c: int | None = None    # shared query latent dim. Defaults to d_model // 2.
    # RMSNorm on queries and keys in the dense baseline. CSA has had this from
    # the start (paper §2.3.3: "avoids exploding attention logits and may
    # improve training stability"); vanilla did not, which handed CSA an
    # arm-asymmetric stability advantage that grows with context. Measured on
    # the pre-fix runs, with --grad-clip 1.0: vanilla's median gradient norm
    # was 10.7 at 8K and 15.4 at 16K (clipped on 91% / 94% of steps, i.e. an
    # effective LR ~1/10 of nominal), while CSA sat at ~0.6 and was essentially
    # never clipped. Default on: this is what makes the baseline fair, and
    # QK-norm is standard for dense attention. Pass --no-vanilla-qk-norm to
    # reproduce the older runs.
    vanilla_qk_norm: bool = True

    csa_top_k: int = 16           # eval-time top-k blocks per query (D3).
    # Paper §2.3.3 "Additional Branch of Sliding Window Attention": each query
    # additionally attends to the `csa_n_win` most recent *uncompressed* tokens,
    # concatenated with the selected compressed entries before a single core
    # attention (Figure 3). Without it a query cannot see any token inside its
    # own compressed block — the paper's stated reason for the branch. 0
    # disables it and reproduces the compressed-branch-only model. DeepSeek-V4
    # uses 128 for both Flash and Pro (§4.2.1), at m = 4.
    csa_n_win: int = 0
    # Query-axis chunk for core attention. 0 = one shot (all n queries).
    # The score matrix is (b, n_heads, n, n_blk) and autocast runs softmax in
    # fp32, so at n=16384/m=4 one layer's scores are 1.6GB; chunking bounds the
    # transient copies (mask + softmax) without changing the result.
    csa_chunk: int = 0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.d_ff is None:
            # 8/3 keeps SwiGLU param count comparable to a 4x GELU MLP.
            self.d_ff = _round_to(int(8 * self.d_model / 3), 64)
        if self.csa_c is None:
            self.csa_c = self.head_dim
        if self.csa_c_i is None:
            self.csa_c_i = max(8, self.csa_c // 2)
        if self.csa_n_h_i is None:
            self.csa_n_h_i = max(2, self.n_heads // 2)
        if self.csa_d_c is None:
            self.csa_d_c = self.d_model // 2
        assert self.block_size % self.csa_m == 0, "block_size must be divisible by csa_m"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def _round_to(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt(mean(x^2) + eps) — standard RMSNorm
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU MLP: (xW_gate ⊙ silu(xW_up)) W_down."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.w_gate(x) * F.silu(self.w_up(x)))


# Finite stand-in for -inf in attention masks. Large enough that exp() of it
# underflows to exactly 0 in fp64/fp32/bf16 (so masked entries carry no weight),
# but finite, so a fully-masked row softmaxes to a uniform distribution instead
# of NaN. Avoiding the NaN lets us drop the nan_to_num repair, which would
# otherwise cost a second full-size copy of the score matrix.
_MASK_FILL = -1e9


class VanillaMultiHeadAttention(nn.Module):
    """Standard causal multi-head self-attention. The Stage-1 baseline."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # Per-head QK normalisation, mirroring CSAAttention's q_norm / k_norm
        # so the two arms differ in attention mechanism and nothing else.
        self.qk_norm = cfg.vanilla_qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim)
            self.k_norm = RMSNorm(cfg.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)            # each: (b, n, h, d_h)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (b, h, n, d_h)
        # PyTorch's SDPA picks flash / mem-efficient kernels when available.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, n, self.n_heads * self.head_dim)
        return self.proj(out)


class CSACompression(nn.Module):
    """Paper eqs. 9-12: compress every m KV tokens into one entry.

    For block i ∈ [0, n/m):
      - "current" window covers token positions [m*i, m*(i+1))    via C^a, Z^a
      - "previous" window covers token positions [m*(i-1), m*i)   via C^b, Z^b
      - At i=0 the previous window is out of range: Z^b is padded with -inf
        (so its softmax weights are zero) and C^b with zeros (so the
        contribution is zero even without the weights).
      - Adjacent compressed entries C_i and C_{i-1} share token *ranges*
        through different projections: C^b for C_i overlaps C^a for C_{i-1}.

    Eq. 11's "Softmax_row" normalizes across the 2m source positions for
    each of the c feature dims independently — this is per-feature pooling,
    not standard attention softmax.

    Output: C^Comp ∈ R^(b, n/m, c).
    """

    def __init__(self, d_model: int, c: int, m: int) -> None:
        super().__init__()
        self.m = m
        self.c = c
        # eq. 9: two KV streams
        self.W_aKV = nn.Linear(d_model, c, bias=False)
        self.W_bKV = nn.Linear(d_model, c, bias=False)
        # eq. 10: two compression-weight streams
        self.W_aZ = nn.Linear(d_model, c, bias=False)
        self.W_bZ = nn.Linear(d_model, c, bias=False)
        # learnable positional biases inside a compressed block (eq. 11)
        self.B_a = nn.Parameter(torch.zeros(m, c))
        self.B_b = nn.Parameter(torch.zeros(m, c))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        m, c = self.m, self.c
        assert n % m == 0, f"sequence length {n} not divisible by m={m}"
        n_blk = n // m

        # eq. 9, 10: per-token streams, all shape (b, n, c)
        C_a = self.W_aKV(h)
        C_b = self.W_bKV(h)
        Z_a = self.W_aZ(h)
        Z_b = self.W_bZ(h)

        # Reshape "current" windows: positions [mi, m(i+1)) for block i.
        C_a_cur = rearrange(C_a, "b (n_blk m) c -> b n_blk m c", m=m)
        Z_a_cur = rearrange(Z_a, "b (n_blk m) c -> b n_blk m c", m=m)

        # "previous" windows: positions [m(i-1), mi). Shift right by m via
        # front-padding with m positions, then reshape. The padded slot for
        # block i=0 corresponds to the [-m, 0) "before-the-sequence" range —
        # Z^b gets -inf (its softmax weight will be 0), C^b gets zeros (so
        # the weighted-sum contribution is 0 even if weights weren't 0).
        # Pad signature for (b, n, c) is (last_dim_l, last_dim_r, ..., n_l, n_r):
        # we want n_l = m, n_r = 0 to add m rows at the front.
        Z_b_padded = F.pad(Z_b, (0, 0, m, 0), value=float("-inf"))
        C_b_padded = F.pad(C_b, (0, 0, m, 0), value=0.0)
        Z_b_prev = rearrange(Z_b_padded[:, :n, :], "b (n_blk m) c -> b n_blk m c", m=m)
        C_b_prev = rearrange(C_b_padded[:, :n, :], "b (n_blk m) c -> b n_blk m c", m=m)

        # eq. 11: stack along block axis (2m positions per block) and softmax
        # PER-FEATURE across those 2m. Bias broadcasts (m, c) -> (1, 1, m, c).
        Z_a_biased = Z_a_cur + self.B_a
        Z_b_biased = Z_b_prev + self.B_b
        Z_concat = torch.cat([Z_a_biased, Z_b_biased], dim=2)  # (b, n_blk, 2m, c)
        S = F.softmax(Z_concat, dim=2)                          # softmax over 2m axis
        S_a, S_b = S[:, :, :m, :], S[:, :, m:, :]

        # eq. 12: weighted sum over the 2m source positions, per feature.
        C_comp = (S_a * C_a_cur).sum(dim=2) + (S_b * C_b_prev).sum(dim=2)
        return C_comp  # (b, n_blk, c)


class CSAIndexer(nn.Module):
    """Paper eqs. 13-17: lightning indexer producing per-(query, block) scores.

      eq. 13:  c^Q_t = h_t · W^DQ                  shape (b, n, d_c)
      eq. 14:  q^I_t = c^Q_t · W^IUQ -> heads      shape (b, n, n_h^I, c^I)
      eq. 15:  w^I_t = h_t · W^w                   shape (b, n, n_h^I)
      K^IComp: separate compressor (D1)            shape (b, n_blk, c^I)
      eq. 16:  I[t, s] = sum_h w^I[t, h] * ReLU(q^I[t, h] · K^IComp[s])

    The shared latent `c^Q` is exposed (returned alongside the scores) so
    Stage 4's core-query path can reuse it via W^UQ.

    Training signal (Stage 5, V3.2 scheme): the indexer is trained by an
    auxiliary KL loss aligning softmax(I_{t,:}) with the L1-normalized
    head-summed dense-attention distribution p_{t,:}. The indexer no
    longer feeds the core attention's logits; it only drives top-k
    selection of compressed blocks (eq. 17, applied at eval and during
    sparse-training phases). The indexer's input is detached in
    CSAIndexer.forward so L_I doesn't push gradients back into the
    upstream main-model path.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.m = cfg.csa_m
        self.c_i = cfg.csa_c_i
        self.n_h_i = cfg.csa_n_h_i
        self.d_c = cfg.csa_d_c
        # eq. 13, 14: the shared latent and its indexer up-projection
        self.W_DQ = nn.Linear(cfg.d_model, cfg.csa_d_c, bias=False)
        self.W_IUQ = nn.Linear(cfg.csa_d_c, cfg.csa_c_i * cfg.csa_n_h_i, bias=False)
        # eq. 15: per-head scoring weights
        self.W_w = nn.Linear(cfg.d_model, cfg.csa_n_h_i, bias=False)
        # K^IComp: separate compressor with its own projections (D1).
        self.compress_indexer = CSACompression(cfg.d_model, cfg.csa_c_i, cfg.csa_m)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # c_q is the shared latent; it MUST flow back to W_DQ from the
        # main-attention path (W_UQ in CSAAttention), so we leave it
        # un-detached on the way out. For the indexer's *local* use of c_q
        # (and h), we detach so L_I doesn't push gradient into W_DQ or
        # upstream. This is the V3.2 "detach the indexer input" trick
        # adapted to the V4 shared-latent design.
        c_q = self.W_DQ(h)                                                  # (b, n, d_c)
        q_i = rearrange(self.W_IUQ(c_q.detach()), "b n (h c) -> b n h c", h=self.n_h_i)
        w_i = self.W_w(h.detach())                                          # (b, n, n_h_i)
        K_iComp = self.compress_indexer(h.detach())                         # (b, n_blk, c_i)
        # eq. 16: per-head ReLU dot product, then per-head weighted sum.
        # einsum: (b, n, h, c_i) x (b, n_blk, c_i) -> (b, n, h, n_blk)
        dots = einsum(q_i, K_iComp, "b n h c, b s c -> b n h s")
        scores = (w_i.unsqueeze(-1) * F.relu(dots)).sum(dim=2)              # (b, n, n_blk)
        return scores, c_q


class CSAAttention(nn.Module):
    """Stage-4 CSA: full Shared-KV MQA over indexer-selected compressed blocks.

    Now matches the paper's eqs. 18-19 verbatim:
        eq. 18  [q_{t,1};...;q_{t,n_h}] = c^Q_t · W^UQ
        eq. 19  o_{t,i} = CoreAttn(q_{t,i}, K=C^SprsComp_t, V=C^SprsComp_t)

    The shared latent c^Q (eq. 13) is computed once inside the indexer and
    feeds BOTH the indexer queries (via W^IUQ) and the core queries (via
    W^UQ here). This collapses the two query paths into a single low-rank
    bottleneck, exactly as the paper specifies, and trims params (the old
    W_Q was d × n_h·c; W^UQ is only d_c × n_h·c, with d_c = d/2).

    Indexer scores `I[t, s]` no longer feed back into the core-attention
    logits — that additive-bias coupling (our Stage-3 D3 invention) is
    replaced by the V3.2-style auxiliary KL loss computed at MiniTransformer
    level. At eval, indexer scores still drive a top-k mask restricting
    core attention to the `csa_top_k` highest-scoring blocks per query.

    Causal mask: query at token t may only attend to compressed blocks
    s < floor(t/m). Positions t < m have no causally-valid block — their
    output is forced to zero by the safe softmax. The training loop masks
    those positions from the loss (per D2).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.c = cfg.csa_c
        self.m = cfg.csa_m
        self.top_k = cfg.csa_top_k
        self.chunk_size = cfg.csa_chunk
        self.n_win = cfg.csa_n_win
        if self.n_win > 0:
            # Figure 3 routes the sliding-window entries straight from the KV
            # hidden states, bypassing the token-level compressor, so this is a
            # projection of its own rather than a reuse of C^a. The paper does
            # not specify it explicitly — see notes.md, D7.
            self.W_win = nn.Linear(cfg.d_model, cfg.csa_c, bias=False)
            self.win_norm = RMSNorm(cfg.csa_c)
        # Toggle for the eval-time top-k filter. Set False to evaluate
        # densely (no top-k), used by the training loop's dual eval to
        # report the train/eval mismatch metric (D3).
        self.eval_apply_topk = True
        self.compress = CSACompression(cfg.d_model, cfg.csa_c, cfg.csa_m)
        self.indexer = CSAIndexer(cfg)
        # eq. 18: up-project the shared latent c^Q into n_h core query heads.
        self.W_UQ = nn.Linear(cfg.csa_d_c, cfg.n_heads * cfg.csa_c, bias=False)
        # paper §2.3.3: per-head RMSNorm before core attention
        self.q_norm = RMSNorm(cfg.csa_c)
        self.k_norm = RMSNorm(cfg.csa_c)
        self.W_O = nn.Linear(cfg.n_heads * cfg.csa_c, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (residual_delta, aux) where aux holds tensors needed by the
        V3.2 indexer KL loss at MiniTransformer level: `p` (teacher target,
        detached) and `I_masked` (indexer scores with causally-invalid blocks
        masked out)."""
        n = x.shape[1]
        m = self.m
        n_blk = n // m

        # Indexer produces both the per-block scores and the shared latent c^Q.
        indexer_scores, c_q = self.indexer(x)

        # eq. 18: core queries from the same latent c^Q the indexer uses.
        q = rearrange(self.W_UQ(c_q), "b n (h c) -> b n h c", h=self.n_heads)
        q = self.q_norm(q)

        # Compressed keys = values, shared across all query heads (MQA).
        kv = self.compress(x)        # (b, n_blk, c)
        kv = self.k_norm(kv)

        # Uncompressed per-token KV entries for the sliding-window branch.
        # Shared across heads like the compressed entries (MQA).
        win = self.win_norm(self.W_win(x)) if self.n_win > 0 else None   # (b, n, c)

        # Causal mask: token index t may attend to blocks s < t // m.
        device = x.device
        token_idx = torch.arange(n, device=device).unsqueeze(1)             # (n, 1)
        block_idx = torch.arange(n_blk, device=device).unsqueeze(0)         # (1, n_blk)
        causal = block_idx < (token_idx // m)                               # (n, n_blk)
        # Positions t < m have no causally-valid block at all. Their rows are
        # fully masked, and _MASK_FILL makes them softmax to uniform rather
        # than NaN; _attend zeroes their output and teacher rows explicitly.
        row_valid = causal.any(dim=-1)                                      # (n,)

        apply_topk = (not self.training) and self.eval_apply_topk and self.top_k < n_blk

        # Chunk over query positions. Purely a memory device: each chunk's
        # score matrix and its masked/softmaxed copies are freed before the
        # next chunk allocates, so peak transient memory scales with
        # chunk_size rather than n. Results are identical to one shot.
        chunk = self.chunk_size if self.chunk_size > 0 else n
        outs: list[torch.Tensor] = []
        ps: list[torch.Tensor] = []
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            # Keys this chunk's sliding windows can reach: the earliest query
            # in the chunk looks back n_win - 1 positions.
            w_lo = max(0, lo - self.n_win + 1) if self.n_win > 0 else 0
            out_c, p_c = self._attend(
                q[:, lo:hi],
                kv,
                causal[lo:hi],
                indexer_scores[:, lo:hi],
                row_valid[lo:hi],
                apply_topk,
                win[:, w_lo:hi] if win is not None else None,
                lo,
                w_lo,
            )
            outs.append(out_c)
            ps.append(p_c)

        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        p = ps[0] if len(ps) == 1 else torch.cat(ps, dim=1)

        aux = {
            "p": p,
            "I_masked": indexer_scores.masked_fill(~causal, _MASK_FILL),
        }
        return self.W_O(out), aux

    def _attend(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        causal: torch.Tensor,
        indexer_scores: torch.Tensor,
        blk_valid: torch.Tensor,
        apply_topk: bool,
        win: torch.Tensor | None = None,
        lo: int = 0,
        w_lo: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core attention for one chunk of query positions.

        Shapes: q (b, n_c, h, c), kv (b, n_blk, c), causal (n_c, n_blk),
        indexer_scores (b, n_c, n_blk), blk_valid (n_c,) marking queries with at
        least one causally-valid compressed block. `win` is the slice of
        uncompressed sliding-window entries this chunk can reach, covering
        absolute positions [w_lo, lo + n_c); None disables the branch.
        Returns (out (b, n_c, h*c), p (b, n_c, n_blk) detached).
        """
        # Core attention logits. Indexer scores no longer feed back as an
        # additive bias — they only drive eval-time top-k selection below.
        scores = einsum(q, kv, "b n h c, b s c -> b h n s") / math.sqrt(self.c)
        scores = scores.masked_fill(~causal, _MASK_FILL)

        # Top-k selection (eval / sparse-training phase). Indexer's
        # causally-masked scores choose which blocks survive; same set
        # applied to all query heads (MQA: all heads share K=V). Indexer
        # scores are ReLU sums (>= 0), so any causally-valid block outranks
        # every masked one and a valid row always keeps >= 1 valid block.
        if apply_topk:
            masked_idx = indexer_scores.masked_fill(~causal, _MASK_FILL)
            _, topk_idx = masked_idx.topk(k=self.top_k, dim=-1)             # (b, n_c, k)
            keep = torch.zeros_like(masked_idx, dtype=torch.bool)
            keep.scatter_(-1, topk_idx, True)                                # (b, n_c, n_blk)
            scores = scores.masked_fill(~keep.unsqueeze(1), _MASK_FILL)

        # Sliding-window branch (paper §2.3.3 / Figure 3). The window entries
        # are concatenated with the selected compressed entries and share a
        # SINGLE core attention, so the model apportions attention mass between
        # local and compressed context rather than summing two attentions.
        n_blk = kv.shape[1]
        any_valid = blk_valid
        if win is not None:
            n_c = q.shape[1]
            t_abs = torch.arange(lo, lo + n_c, device=q.device).unsqueeze(1)
            s_abs = torch.arange(w_lo, lo + n_c, device=q.device).unsqueeze(0)
            # Causal and within the last n_win tokens, inclusive of t itself.
            win_ok = (s_abs <= t_abs) & (s_abs > t_abs - self.n_win)
            scores_win = einsum(q, win, "b n h c, b s c -> b h n s") / math.sqrt(self.c)
            scores_win = scores_win.masked_fill(~win_ok, _MASK_FILL)
            scores = torch.cat([scores, scores_win], dim=-1)
            # Every position has at least itself in the window, so no row is
            # fully masked once the branch is on.
            any_valid = blk_valid | win_ok.any(dim=-1)

        attn = F.softmax(scores, dim=-1)
        attn_blk = attn[..., :n_blk]

        # V3.2 eq. 3: teacher target p_{t,:} = L1-normalize_seq(sum_h attn).
        # Detached so L_I doesn't push gradient back into the attention path
        # (we want the indexer to chase attention, not the other way around).
        # Only the compressed-block share is used: the indexer scores blocks,
        # not window tokens, so renormalizing over blocks keeps p a
        # distribution over exactly what the indexer has to predict.
        p = attn_blk.sum(dim=1)                                              # (b, n_c, n_blk)
        p_sum = p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        p = (p / p_sum).detach()

        out = einsum(attn_blk, kv, "b h n s, b s c -> b h n c")
        if win is not None:
            out = out + einsum(attn[..., n_blk:], win, "b h n s, b s c -> b h n c")
        out = rearrange(out, "b h n c -> b n (h c)")

        # Rows with nothing to attend to softmaxed to uniform over an all-masked
        # row; force their output to zero. Without the window branch that is
        # every t < m; with it, no row at all.
        out = out * any_valid.view(1, -1, 1).to(out.dtype)
        # The teacher is zeroed wherever no *block* is valid, independently of
        # the window: at t < m the indexer has nothing to predict even though
        # the window now gives those positions real attention output.
        p.masked_fill_(~blk_valid.view(1, -1, 1), 0.0)                       # p is detached
        return out, p


def _build_attention(cfg: ModelConfig) -> nn.Module:
    if cfg.attention == "vanilla":
        return VanillaMultiHeadAttention(cfg)
    if cfg.attention == "csa":
        return CSAAttention(cfg)
    raise ValueError(f"unknown attention type: {cfg.attention!r}")


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = _build_attention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Returns (x, aux). aux is the CSA layer's indexer-KL inputs, or None
        for vanilla attention."""
        h = self.norm_attn(x)
        if isinstance(self.attn, CSAAttention):
            attn_out, aux = self.attn(h)
        else:
            attn_out = self.attn(h)
            aux = None
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x, aux


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class MiniTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tied weights: lm_head shares the token-embedding matrix.
        self.lm_head.weight = self.tok_embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Returns (logits, lm_loss, indexer_loss).

        `indexer_loss` is the V3.2 auxiliary KL loss (eq. 3) summed across
        CSA layers and averaged over valid positions; None when no CSA
        layers exist or when targets is None. The two losses are kept
        separate so the trainer can implement the V3.2 two-phase schedule:
        phase-1 indexer-only backward, phase-2 joint backward.
        """
        b, n = idx.shape
        assert n <= self.cfg.block_size, f"sequence length {n} > block_size {self.cfg.block_size}"
        pos = torch.arange(n, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        auxes: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            x, aux = block(x)
            if aux is not None:
                auxes.append(aux)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        lm_loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view(b, n)
            if loss_mask is not None:
                lm_loss = (ce * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            else:
                lm_loss = ce.mean()

        indexer_loss = None
        if auxes and targets is not None:
            indexer_loss = self._indexer_kl_loss(auxes, loss_mask)
        return logits, lm_loss, indexer_loss

    def _indexer_kl_loss(
        self,
        auxes: list[dict[str, torch.Tensor]],
        loss_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """V3.2 eq. 3: L^I = sum_t D_KL(p_{t,:} || softmax(I_{t,:})),
        averaged across CSA layers and valid query positions.

        `p` is detached at the CSAAttention level. We mask invalid (t<m)
        positions out so they contribute zero gradient to the indexer.
        """
        total = 0.0
        for aux in auxes:
            p = aux["p"]              # (b, n, n_blk); zero rows for t < m
            I_m = aux["I_masked"]     # (b, n, n_blk); _MASK_FILL on invalid blocks

            # Expanded form of D_KL(p || softmax(I)), which avoids ever
            # materializing log_softmax(I):
            #
            #   KL = sum_s p_s log p_s - sum_s p_s log q_s
            #      = sum_s p_s log p_s - sum_s p_s I_s + logsumexp(I) * sum_s p_s
            #
            # The log_softmax output is a full (b, n, n_blk) fp32 tensor that
            # autograd must keep for backward (1.2GB per layer at n=16384);
            # `logsumexp` keeps only its (b, n) output instead. Algebraically
            # identical, and exact rather than approximate.
            #
            # This relies on I_m being masked with a large *finite* value: with
            # -inf, the p_s * I_s product at masked blocks would be 0 * -inf =
            # NaN rather than 0.
            #
            # The entropy term depends only on the detached teacher, so it is
            # computed under no_grad and adds nothing to the graph. xlogy gives
            # the 0 log 0 = 0 convention directly.
            with torch.no_grad():
                neg_entropy = torch.xlogy(p, p).sum(dim=-1)     # (b, n)
            # sum_s p_s is 1 on valid rows and exactly 0 on fully-masked ones
            # (t < m), which zeroes those rows' KL as the -inf form did.
            p_mass = p.sum(dim=-1)                              # (b, n)
            cross = (p * I_m).sum(dim=-1) - torch.logsumexp(I_m, dim=-1) * p_mass
            kl_per_pos = neg_entropy - cross                    # (b, n)

            if loss_mask is not None:
                kl_per_pos = kl_per_pos * loss_mask
                norm = loss_mask.sum().clamp_min(1.0)
            else:
                norm = torch.tensor(kl_per_pos.numel(), device=kl_per_pos.device)
            total = total + kl_per_pos.sum() / norm
        return total / len(auxes)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        # CSA requires sequence length divisible by m. Front-pad the context
        # if needed; the pad slots fall in the loss-masked [0, m) region the
        # model produces zero output for, so they don't perturb anything that
        # matters. We only ever read logits at the last position.
        m_div = self.cfg.csa_m if self.cfg.attention == "csa" else 1
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            pad_len = (-idx_cond.shape[1]) % m_div
            if pad_len:
                pad = idx_cond.new_zeros(idx_cond.shape[0], pad_len)
                idx_cond = torch.cat([pad, idx_cond], dim=1)
            logits, _, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            # tied weights => only subtract once
            n -= self.tok_embed.weight.numel()
            n -= self.pos_embed.weight.numel()
        return n
