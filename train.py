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
    dataset: str = "tinyshakespeare"   # "tinyshakespeare" | "enwiki8"

    # model
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    block_size: int = 1024
    csa_m: int = 4
    csa_c: int = 0          # 0 -> head_dim
    csa_c_i: int = 0        # 0 -> csa_c // 2
    csa_n_h_i: int = 0      # 0 -> max(2, n_heads // 2)
    csa_d_c: int = 0        # 0 -> d_model // 2
    csa_top_k: int = 16     # eval-time top-k blocks per query

    # V3.2-style two-phase training (CSA only)
    indexer_warmup_iters: int = 0  # phase 1 (indexer-only). 0 disables warmup.

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


def _set_csa_topk(m: torch.nn.Module, enabled: bool) -> None:
    """Toggle the eval-time top-k filter on every CSAAttention block."""
    for module in m.modules():
        if isinstance(module, model.CSAAttention):
            module.eval_apply_topk = enabled


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

    # For CSA, run two eval passes per D3: dense (no top-k) and top-k.
    # Dense matches the training regime; top-k matches the deployment regime.
    # Use the same batches for both so the comparison is paired.
    is_csa = cfg.attention == "csa"
    out: dict[str, float] = {}
    for split_name, split_data in (("train", ds.train), ("val", ds.val)):
        dense_losses = torch.zeros(cfg.eval_iters)
        topk_losses = torch.zeros(cfg.eval_iters) if is_csa else None
        indexer_losses = torch.zeros(cfg.eval_iters) if is_csa else None
        for i in range(cfg.eval_iters):
            x, y = data.get_batch(split_data, cfg.block_size, cfg.batch_size, device)
            if is_csa:
                _set_csa_topk(m, False)
                _, lm_loss, idx_loss = m(x, y, loss_mask=loss_mask)
                dense_losses[i] = lm_loss.item()
                indexer_losses[i] = idx_loss.item()
                _set_csa_topk(m, True)
                _, lm_loss, _ = m(x, y, loss_mask=loss_mask)
                topk_losses[i] = lm_loss.item()
            else:
                _, lm_loss, _ = m(x, y, loss_mask=loss_mask)
                dense_losses[i] = lm_loss.item()
        # Primary metric: dense (matches training regime). Top-k surfaces
        # the train/eval gap when CSA is deployed sparsely.
        out[f"{split_name}_loss"] = dense_losses.mean().item()
        out[f"{split_name}_ppl"] = math.exp(dense_losses.mean().item())
        if is_csa:
            out[f"{split_name}_loss_topk"] = topk_losses.mean().item()
            out[f"{split_name}_ppl_topk"] = math.exp(topk_losses.mean().item())
            out[f"{split_name}_indexer_kl"] = indexer_losses.mean().item()
    if is_csa:
        _set_csa_topk(m, True)  # restore default
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

    ds = data.load(cfg.dataset)
    test_str = f", test={len(ds.test):,}" if ds.test is not None else ""
    print(
        f"[data]  {cfg.dataset}  vocab={ds.vocab_size}, "
        f"train={len(ds.train):,} tokens, val={len(ds.val):,}{test_str}"
    )

    mcfg = model.ModelConfig(
        vocab_size=ds.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        block_size=cfg.block_size,
        attention=cfg.attention,
        csa_m=cfg.csa_m,
        csa_c=cfg.csa_c if cfg.csa_c > 0 else None,
        csa_c_i=cfg.csa_c_i if cfg.csa_c_i > 0 else None,
        csa_n_h_i=cfg.csa_n_h_i if cfg.csa_n_h_i > 0 else None,
        csa_d_c=cfg.csa_d_c if cfg.csa_d_c > 0 else None,
        csa_top_k=cfg.csa_top_k,
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
            line = (
                f"[eval]  step {step:5d}  "
                f"train {metrics['train_loss']:.4f} (ppl {metrics['train_ppl']:.2f})  "
                f"val {metrics['val_loss']:.4f} (ppl {metrics['val_ppl']:.2f})"
            )
            if "val_loss_topk" in metrics:
                line += f"  val(topk) {metrics['val_loss_topk']:.4f} (ppl {metrics['val_ppl_topk']:.2f})"
            if "val_indexer_kl" in metrics:
                line += f"  L_I {metrics['val_indexer_kl']:.4f}"
            print(line)

        if step == cfg.max_iters:
            break

        # train step
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = data.get_batch(ds.train, cfg.block_size, cfg.batch_size, device)
        _, lm_loss, indexer_loss = m(x, y, loss_mask=train_loss_mask)
        # V3.2 two-phase schedule:
        #   Phase 1 (warmup, step < indexer_warmup_iters): backward only L_I,
        #     so only indexer params receive gradient. Main model is "frozen"
        #     by never receiving a gradient (AdamW skips params with .grad=None).
        #   Phase 2 (joint): backward L_LM + L_I; the detach in CSAIndexer.forward
        #     keeps the two flows isolated to their respective params.
        # For vanilla attention there's no indexer, so we always backward L_LM.
        in_warmup = (indexer_loss is not None) and (step < cfg.indexer_warmup_iters)
        if in_warmup:
            loss = indexer_loss
        elif indexer_loss is not None:
            loss = lm_loss + indexer_loss
        else:
            loss = lm_loss

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
                "loss": lm_loss.item(),
                "indexer_loss": indexer_loss.item() if indexer_loss is not None else None,
                "phase": 1 if in_warmup else 2,
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_s": tok_per_s,
                "tokens_seen": tokens_seen,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            extra = ""
            if indexer_loss is not None:
                extra = f"  L_I {indexer_loss.item():.4f}  phase {1 if in_warmup else 2}"
            print(
                f"[train] step {step:5d}  loss {lm_loss.item():.4f}{extra}  "
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
