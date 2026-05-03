"""Decoder-only transformer for the mini-deepseek-v4 study.

Stage 1: vanilla multi-head causal self-attention only. The attention block
is the single component that will be swapped for CSA in Stage 2; everything
else (embeddings, RMSNorm, SwiGLU MLP, output head) is shared between the
two architectures so the comparison isolates the attention mechanism.

Per design decision D4 (notes.md): no RoPE — learned absolute position
embeddings, identical for both architectures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    attention: str = "vanilla"  # "vanilla" | "csa" (csa implemented in Stage 2)
    d_ff: int | None = None     # SwiGLU inner dim. Defaults to ~8/3 * d_model.

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.d_ff is None:
            # 8/3 keeps SwiGLU param count comparable to a 4x GELU MLP.
            self.d_ff = _round_to(int(8 * self.d_model / 3), 64)

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


def _build_attention(cfg: ModelConfig) -> nn.Module:
    if cfg.attention == "vanilla":
        return VanillaMultiHeadAttention(cfg)
    if cfg.attention == "csa":
        raise NotImplementedError("CSA arrives in Stage 2 — not implemented yet.")
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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
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
