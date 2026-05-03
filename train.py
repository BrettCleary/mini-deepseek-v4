"""Training loop for the mini-deepseek-v4 study.

Usage:
    python train.py --attention vanilla --max-iters 3000 --run-name baseline-v1
    python train.py --attention csa     --max-iters 3000 --run-name csa-v1   # Stage 2+
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import data
import model

RUNS_DIR = Path(__file__).parent / "runs"


# ---------------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    attention: str = "vanilla"
    run_name: str = "baseline"

    # model
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    block_size: int = 1024
    csa_m: int = 4
    csa_c: int = 0  # 0 -> default to head_dim (passed as None to ModelConfig)

    # optim
    batch_size: int = 32
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 50
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # misc
    seed: int = 1337
    compile: bool = False  # torch.compile; off by default for clarity in v1


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    cfg = TrainConfig()
    for f in cfg.__dataclass_fields__.values():
        default = getattr(cfg, f.name)
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(default, bool):
            p.add_argument(flag, action="store_true", default=default)
        else:
            p.add_argument(flag, default=default, type=type(default))
    ns = p.parse_args()
    return TrainConfig(**{k: getattr(ns, k) for k in cfg.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Cosine schedule with linear warmup."""
    if step < cfg.warmup_iters:
        return cfg.lr * (step + 1) / cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


def make_loss_mask(
    batch_size: int, block_size: int, mask_first: int, device: torch.device
) -> torch.Tensor | None:
    """Per-design-decision D2: zero out the first `mask_first` positions of
    every sequence so they don't contribute to the loss. CSA produces zero
    attention output for those positions (no causally-valid block exists),
    so they have no learning signal and shouldn't move the gradient."""
    if mask_first <= 0:
        return None
    mask = torch.ones((batch_size, block_size), device=device)
    mask[:, :mask_first] = 0.0
    return mask


@torch.no_grad()
def evaluate(
    m: model.MiniTransformer,
    ds: data.CharDataset,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    m.eval()
    # Mask first csa_m positions for csa eval (those positions have no
    # learning signal — see D2). For vanilla, no mask: every position is
    # valid. The 0.4% gap (4/1024) between vanilla "all positions" and csa
    # "excluding first m" is documented in the README as negligible.
    mask_first = cfg.csa_m if cfg.attention == "csa" else 0
    loss_mask = make_loss_mask(cfg.batch_size, cfg.block_size, mask_first, device)
    out: dict[str, float] = {}
    for split_name, split_data in (("train", ds.train), ("val", ds.val)):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = data.get_batch(split_data, cfg.block_size, cfg.batch_size, device)
            _, loss = m(x, y, loss_mask=loss_mask)
            losses[i] = loss.item()
        out[f"{split_name}_loss"] = losses.mean().item()
        out[f"{split_name}_ppl"] = math.exp(losses.mean().item())
    m.train()
    return out


def make_param_groups(m: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Standard split: 2D tensors get weight decay, 1D (norms, biases, embeddings) don't."""
    decay, no_decay = [], []
    for p in m.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
        torch.set_float32_matmul_precision("high")

    print(f"[setup] device={device}")

    ds = data.load()
    print(f"[data]  vocab={ds.vocab_size}, train={len(ds.train):,} tokens, val={len(ds.val):,}")

    mcfg = model.ModelConfig(
        vocab_size=ds.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        block_size=cfg.block_size,
        attention=cfg.attention,
        csa_m=cfg.csa_m,
        csa_c=cfg.csa_c if cfg.csa_c > 0 else None,
    )
    m = model.MiniTransformer(mcfg).to(device)
    if cfg.compile:
        m = torch.compile(m)
    print(f"[model] {cfg.attention}, params={m.num_params():,} (excl emb={m.num_params(exclude_embeddings=True):,})")

    optimizer = torch.optim.AdamW(
        make_param_groups(m, cfg.weight_decay),
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )

    run_dir = RUNS_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"train": asdict(cfg), "model": asdict(mcfg)}, indent=2)
    )
    log_path = run_dir / "log.jsonl"
    log_f = log_path.open("w")
    print(f"[run]   logging to {run_dir}")

    train_mask_first = cfg.csa_m if cfg.attention == "csa" else 0
    train_loss_mask = make_loss_mask(cfg.batch_size, cfg.block_size, train_mask_first, device)

    m.train()
    t0 = time.time()
    tokens_seen = 0
    step_t0 = time.time()
    last_log_step = 0

    for step in range(cfg.max_iters + 1):
        # eval
        if step % cfg.eval_interval == 0:
            metrics = evaluate(m, ds, cfg, device)
            elapsed = time.time() - t0
            entry = {"step": step, "kind": "eval", "elapsed_s": elapsed, **metrics}
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(
                f"[eval]  step {step:5d}  "
                f"train {metrics['train_loss']:.4f} (ppl {metrics['train_ppl']:.2f})  "
                f"val {metrics['val_loss']:.4f} (ppl {metrics['val_ppl']:.2f})"
            )

        if step == cfg.max_iters:
            break

        # train step
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = data.get_batch(ds.train, cfg.block_size, cfg.batch_size, device)
        _, loss = m(x, y, loss_mask=train_loss_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
        optimizer.step()

        tokens_seen += x.numel()

        if step % 50 == 0 and step > 0:
            dt = time.time() - step_t0
            tok_per_s = (cfg.batch_size * cfg.block_size * (step - last_log_step)) / dt
            entry = {
                "step": step,
                "kind": "train",
                "loss": loss.item(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_s": tok_per_s,
                "tokens_seen": tokens_seen,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(
                f"[train] step {step:5d}  loss {loss.item():.4f}  "
                f"lr {lr:.2e}  grad {grad_norm.item():.2f}  "
                f"tok/s {tok_per_s:,.0f}"
            )
            step_t0 = time.time()
            last_log_step = step

    log_f.close()
    # save final checkpoint (small enough to be useful for downstream Stage 2 cmp)
    torch.save({"model_state": m.state_dict(), "config": asdict(mcfg)}, run_dir / "final.pt")
    print(f"[done] saved {run_dir / 'final.pt'}  total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    train(parse_args())
