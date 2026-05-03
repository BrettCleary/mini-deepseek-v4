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
    csa_m: int = 4              # compression factor: m KV tokens -> 1 compressed entry
    csa_c: int | None = None    # compressed/per-head dim. Defaults to head_dim.

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.d_ff is None:
            # 8/3 keeps SwiGLU param count comparable to a 4x GELU MLP.
            self.d_ff = _round_to(int(8 * self.d_model / 3), 64)
        if self.csa_c is None:
            self.csa_c = self.head_dim
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


class VanillaMultiHeadAttention(nn.Module):
    """Standard causal multi-head self-attention. The Stage-1 baseline."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)            # each: (b, n, h, d_h)
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


class CSAAttention(nn.Module):
    """Stage-2 CSA: compression + dense attention over all C^Comp blocks.

    No lightning indexer (Stage 3 adds it) and no latent-query factorization
    (Stage 4 reuses the indexer's latent c^Q). For Stage 2 we project H
    directly to per-head queries.

    Causal mask: query at token t may only attend to compressed blocks
    s < floor(t/m). Positions t < m have no causally-valid block — their
    output is forced to zero by the safe softmax (rows with all -inf give
    weight 0 everywhere). The training loop masks those positions out of
    the loss (per design decision D2).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.c = cfg.csa_c
        self.m = cfg.csa_m
        self.compress = CSACompression(cfg.d_model, cfg.csa_c, cfg.csa_m)
        # Stage 2 query projection: H -> n_h heads of c. Stage 3+ replaces
        # this with the W^DQ -> W^UQ low-rank pair shared with the indexer.
        self.W_Q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.csa_c, bias=False)
        # paper §2.3.3: per-head RMSNorm before core attention
        self.q_norm = RMSNorm(cfg.csa_c)
        self.k_norm = RMSNorm(cfg.csa_c)
        self.W_O = nn.Linear(cfg.n_heads * cfg.csa_c, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        m, c = self.m, self.c
        n_blk = n // m

        # Queries — per-head, RMSNorm applied over last dim (c) so each head
        # is normalized independently.
        q = rearrange(self.W_Q(x), "b n (h c) -> b n h c", h=self.n_heads)
        q = self.q_norm(q)

        # Compressed keys = values, shared across all query heads (MQA).
        kv = self.compress(x)        # (b, n_blk, c)
        kv = self.k_norm(kv)

        # Scores: each query head h dots its (b, n, c) against the shared
        # (b, n_blk, c). Result (b, h, n, n_blk).
        scores = einsum(q, kv, "b n h c, b s c -> b h n s") / math.sqrt(c)

        # Causal mask: token index t may attend to blocks s < t // m.
        device = x.device
        token_idx = torch.arange(n, device=device).unsqueeze(1)        # (n, 1)
        block_idx = torch.arange(n_blk, device=device).unsqueeze(0)    # (1, n_blk)
        mask = block_idx < (token_idx // m)                            # (n, n_blk)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Safe softmax: for positions where every block is masked (rows where
        # t // m == 0, i.e. t < m), softmax gives NaN. nan_to_num maps those
        # to 0 so the output is exactly zero for those positions.
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = einsum(attn, kv, "b h n s, b s c -> b h n c")
        out = rearrange(out, "b h n c -> b n (h c)")
        return self.W_O(out)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, n = idx.shape
        assert n <= self.cfg.block_size, f"sequence length {n} > block_size {self.cfg.block_size}"
        pos = torch.arange(n, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view(b, n)
            if loss_mask is not None:
                loss = (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            else:
                loss = loss.mean()
        return logits, loss

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
            logits, _ = self.forward(idx_cond)
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
