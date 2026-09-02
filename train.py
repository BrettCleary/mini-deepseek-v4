"""Training loop for the mini-deepseek-v4 study.

Usage:
    python train.py --attention vanilla --max-iters 3000 --run-name baseline-v1
    python train.py --attention csa     --max-iters 3000 --run-name csa-v1   # Stage 2+
"""
from __future__ import annotations

import argparse
import contextlib
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
    # Sliding-window branch (paper 2.3.3 / Fig 3): each query also attends to
    # the csa_n_win most recent uncompressed tokens. 0 disables it. DeepSeek-V4
    # uses 128 at m=4 (4.2.1). Without it a query cannot see any token inside
    # its own compressed block, which costs ~0.5 bpc at byte level.
    csa_n_win: int = 0
    # Query-axis chunk for CSA core attention. 0 = one shot. Pure memory
    # knob: results are unchanged, peak transient memory scales with the
    # chunk instead of block_size. Needed at 16K.
    csa_chunk: int = 0

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
    # LR schedule shape. "cosine" anneals lr -> min_lr over `lr_horizon`;
    # "wsd" holds `lr` flat then decays linearly to `min_lr` over the last
    # `wsd_decay_frac` of `lr_horizon`.
    lr_schedule: str = "cosine"
    # Horizon the LR anneal is measured against. 0 -> fall back to max_iters
    # (the pre-v4 behaviour). Set this explicitly to decouple the anneal from
    # the stopping cap: with early stopping, a run that halts before max_iters
    # never finishes its anneal, so two runs with different caps get different
    # effective schedules even at the same best step. See README "Why: the LR
    # schedule is tied to the iteration cap".
    lr_horizon: int = 0
    wsd_decay_frac: float = 0.2
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1   # accumulate gradients over this many micro-batches per opt step

    # mixed precision
    amp: str = "none"           # "none" | "bf16"  (bf16 halves activation memory)

    # early stop: stop when val_loss hasn't improved for `early_stop_patience`
    # consecutive evals (after LR warmup). 0 disables. Saves best.pt at the
    # iter that achieved the lowest val_loss.
    early_stop_patience: int = 0

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


def lr_horizon_of(cfg: TrainConfig) -> int:
    """Steps the LR anneal is stretched over. Deliberately separate from
    `max_iters`, which is only the stopping backstop."""
    return cfg.lr_horizon if cfg.lr_horizon > 0 else cfg.max_iters


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine or WSD decay over `lr_horizon_of(cfg)`."""
    if step < cfg.warmup_iters:
        return cfg.lr * (step + 1) / cfg.warmup_iters

    horizon = lr_horizon_of(cfg)
    progress = (step - cfg.warmup_iters) / max(1, horizon - cfg.warmup_iters)
    progress = min(max(progress, 0.0), 1.0)

    if cfg.lr_schedule == "cosine":
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))
    if cfg.lr_schedule == "wsd":
        # Flat at `lr` through the stable phase, then linear to `min_lr`.
        stable = 1.0 - cfg.wsd_decay_frac
        if progress <= stable:
            return cfg.lr
        decay = (progress - stable) / max(1e-9, cfg.wsd_decay_frac)
        return cfg.lr + (cfg.min_lr - cfg.lr) * decay
    raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule!r}")


def masked_prefix(cfg: TrainConfig) -> int:
    """Positions excluded from the loss (design decision D2).

    A CSA query at t < m has no causally-valid compressed block, so it gets no
    gradient and must be excluded. The sliding-window branch gives those
    positions real attention output, so with csa_n_win > 0 nothing needs
    masking — which also makes CSA score on exactly the same positions as
    vanilla, removing a small arm-to-arm asymmetry.
    """
    if cfg.attention != "csa" or cfg.csa_n_win > 0:
        return 0
    return cfg.csa_m


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


def _amp_ctx(cfg: TrainConfig, device: torch.device):
    """Return an autocast context for forward+loss computation.

    bf16 halves activation memory and ~2x throughput on Ampere+. softmax
    and layer-norm stay in fp32 under autocast (PyTorch's default cast
    policy), so the safe-softmax + nan_to_num path in CSAAttention is
    unaffected. Gradient clipping and the optimizer step run on
    fp32 master weights regardless.
    """
    if cfg.amp == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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
    mask_first = masked_prefix(cfg)
    loss_mask = make_loss_mask(cfg.batch_size, cfg.block_size, mask_first, device)

    # For CSA, run two eval passes per D3: dense (no top-k) and top-k.
    # Dense matches the training regime; top-k matches the deployment regime.
    # Use the same batches for both so the comparison is paired.
    is_csa = cfg.attention == "csa"
    amp_ctx = _amp_ctx(cfg, device)
    out: dict[str, float] = {}
    for split_name, split_data in (("train", ds.train), ("val", ds.val)):
        dense_losses = torch.zeros(cfg.eval_iters)
        topk_losses = torch.zeros(cfg.eval_iters) if is_csa else None
        indexer_losses = torch.zeros(cfg.eval_iters) if is_csa else None
        for i in range(cfg.eval_iters):
            x, y = data.get_batch(split_data, cfg.block_size, cfg.batch_size, device)
            with amp_ctx:
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
        avg_dense = dense_losses.mean().item()
        out[f"{split_name}_loss"] = avg_dense
        out[f"{split_name}_ppl"] = math.exp(avg_dense)
        out[f"{split_name}_bpc"] = avg_dense / math.log(2)
        if is_csa:
            avg_topk = topk_losses.mean().item()
            out[f"{split_name}_loss_topk"] = avg_topk
            out[f"{split_name}_ppl_topk"] = math.exp(avg_topk)
            out[f"{split_name}_bpc_topk"] = avg_topk / math.log(2)
            out[f"{split_name}_indexer_kl"] = indexer_losses.mean().item()
    if is_csa:
        _set_csa_topk(m, True)  # restore default
    m.train()
    return out


@torch.no_grad()
def evaluate_split_full(
    m: model.MiniTransformer,
    split_data: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Deterministic full sweep over non-overlapping `block_size`-length windows
    of `split_data`. Used for the canonical end-of-run numbers on val and test.

    Accumulates per-batch loss weighted by the number of valid (loss-masked)
    positions, so the reported mean is exactly the per-position log-loss over
    the swept positions. bpc = loss / ln(2) is the standard enwiki8 metric.

    For CSA, the first `csa_m` positions of every window are excluded (no
    causally-valid block exists — D2). Vanilla scores every position.
    """
    m.eval()
    is_csa = cfg.attention == "csa"
    mask_first = masked_prefix(cfg)
    B = cfg.batch_size
    L = cfg.block_size
    amp_ctx = _amp_ctx(cfg, device)

    n = split_data.numel()
    # Each window needs L+1 tokens (x = [i:i+L], y = [i+1:i+L+1]).
    starts = list(range(0, n - L - 1, L))  # non-overlapping; drops last partial

    total_loss_dense = 0.0
    total_loss_topk = 0.0
    total_positions = 0

    for batch_idx in range(0, len(starts), B):
        batch_starts = starts[batch_idx : batch_idx + B]
        bs = len(batch_starts)
        x = torch.stack([split_data[s : s + L] for s in batch_starts]).to(device, non_blocking=True)
        y = torch.stack([split_data[s + 1 : s + L + 1] for s in batch_starts]).to(device, non_blocking=True)

        if mask_first > 0:
            loss_mask = torch.ones((bs, L), device=device)
            loss_mask[:, :mask_first] = 0.0
            valid_pos = bs * (L - mask_first)
        else:
            loss_mask = None
            valid_pos = bs * L

        with amp_ctx:
            if is_csa:
                _set_csa_topk(m, False)
                _, lm_loss_dense, _ = m(x, y, loss_mask=loss_mask)
                _set_csa_topk(m, True)
                _, lm_loss_topk, _ = m(x, y, loss_mask=loss_mask)
                total_loss_dense += lm_loss_dense.item() * valid_pos
                total_loss_topk += lm_loss_topk.item() * valid_pos
            else:
                _, lm_loss_dense, _ = m(x, y, loss_mask=loss_mask)
                total_loss_dense += lm_loss_dense.item() * valid_pos
        total_positions += valid_pos

    if is_csa:
        _set_csa_topk(m, True)
    m.train()

    avg_dense = total_loss_dense / max(1, total_positions)
    out = {
        "loss": avg_dense,
        "ppl": math.exp(avg_dense),
        "bpc": avg_dense / math.log(2),
        "n_positions": total_positions,
        "n_windows": len(starts),
    }
    if is_csa:
        avg_topk = total_loss_topk / max(1, total_positions)
        out["loss_topk"] = avg_topk
        out["ppl_topk"] = math.exp(avg_topk)
        out["bpc_topk"] = avg_topk / math.log(2)
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
        csa_chunk=cfg.csa_chunk,
        csa_n_win=cfg.csa_n_win,
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

    train_mask_first = masked_prefix(cfg)
    train_loss_mask = make_loss_mask(cfg.batch_size, cfg.block_size, train_mask_first, device)

    amp_ctx = _amp_ctx(cfg, device)
    print(
        f"[opt]   amp={cfg.amp}, grad_accum_steps={cfg.grad_accum_steps}, "
        f"effective_batch={cfg.batch_size * cfg.grad_accum_steps}"
    )
    horizon = lr_horizon_of(cfg)
    print(
        f"[lr]    {cfg.lr_schedule}, warmup={cfg.warmup_iters}, horizon={horizon}"
        f"{' (= max_iters)' if cfg.lr_horizon <= 0 else ''}, "
        f"{cfg.lr:.2e} -> {cfg.min_lr:.2e}"
    )
    if cfg.lr_horizon <= 0 and cfg.early_stop_patience > 0:
        print(
            "[lr]    WARNING: early stopping is on and lr_horizon is unset, so the "
            "anneal is tied to max_iters and a run that stops early never completes "
            "it. Cells with different caps are not comparable. Pass --lr-horizon."
        )

    m.train()
    t0 = time.time()
    tokens_seen = 0
    step_t0 = time.time()
    last_log_step = 0
    best_val = float("inf")
    best_step = 0
    evals_since_best = 0
    stopped_early = False

    for step in range(cfg.max_iters + 1):
        # eval
        if step % cfg.eval_interval == 0:
            metrics = evaluate(m, ds, cfg, device)
            elapsed = time.time() - t0
            val_loss = metrics["val_loss"]
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                evals_since_best = 0
                torch.save(
                    {"model_state": m.state_dict(), "config": asdict(mcfg), "step": step, "val_loss": val_loss},
                    run_dir / "best.pt",
                )
            else:
                evals_since_best += 1
            entry = {
                "step": step,
                "kind": "eval",
                "elapsed_s": elapsed,
                "best_val": best_val,
                "best_step": best_step,
                "evals_since_best": evals_since_best,
                **metrics,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            line = (
                f"[eval]  step {step:5d}  "
                f"train {metrics['train_loss']:.4f} (ppl {metrics['train_ppl']:.2f})  "
                f"val {val_loss:.4f} (ppl {metrics['val_ppl']:.2f})"
            )
            if "val_loss_topk" in metrics:
                line += f"  val(topk) {metrics['val_loss_topk']:.4f} (ppl {metrics['val_ppl_topk']:.2f})"
            if "val_indexer_kl" in metrics:
                line += f"  L_I {metrics['val_indexer_kl']:.4f}"
            line += f"  best {best_val:.4f}@{best_step} ({evals_since_best}/{cfg.early_stop_patience or '-'})"
            print(line)

            # Early stop: only arm after warmup so the high-LR ramp doesn't trip it.
            if (
                cfg.early_stop_patience > 0
                and step >= cfg.warmup_iters
                and evals_since_best >= cfg.early_stop_patience
            ):
                stop_entry = {
                    "step": step,
                    "kind": "early_stop",
                    "elapsed_s": elapsed,
                    "best_val": best_val,
                    "best_step": best_step,
                    "patience": cfg.early_stop_patience,
                }
                log_f.write(json.dumps(stop_entry) + "\n")
                log_f.flush()
                print(
                    f"[stop]  early-stop at step {step}: val hasn't improved "
                    f"in {evals_since_best} evals (best {best_val:.4f}@{best_step})"
                )
                stopped_early = True
                break

        if step == cfg.max_iters:
            break

        # train step — one optimizer step accumulating grad_accum_steps micro-batches.
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        in_warmup = (cfg.attention == "csa") and (step < cfg.indexer_warmup_iters)
        optimizer.zero_grad(set_to_none=True)

        sum_lm = 0.0
        sum_indexer = 0.0
        n_indexer = 0
        for _micro in range(cfg.grad_accum_steps):
            x, y = data.get_batch(ds.train, cfg.block_size, cfg.batch_size, device)
            with amp_ctx:
                _, lm_loss, indexer_loss = m(x, y, loss_mask=train_loss_mask)
                # V3.2 two-phase schedule:
                #   Phase 1 (warmup): backward only L_I, so only indexer params
                #     receive gradient (main model is "frozen" via grad=None,
                #     which AdamW skips).
                #   Phase 2 (joint): backward L_LM + L_I; the detach in
                #     CSAIndexer.forward keeps the gradient flows isolated.
                # Vanilla attention has no indexer, so we always backward L_LM.
                if in_warmup:
                    loss = indexer_loss
                elif indexer_loss is not None:
                    loss = lm_loss + indexer_loss
                else:
                    loss = lm_loss
                loss = loss / cfg.grad_accum_steps
            loss.backward()

            sum_lm += lm_loss.item()
            if indexer_loss is not None:
                sum_indexer += indexer_loss.item()
                n_indexer += 1
            tokens_seen += x.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
        optimizer.step()

        avg_lm = sum_lm / cfg.grad_accum_steps
        avg_indexer = (sum_indexer / n_indexer) if n_indexer > 0 else None

        if step % 50 == 0 and step > 0:
            dt = time.time() - step_t0
            steps_done = step - last_log_step
            tok_per_s = (cfg.batch_size * cfg.grad_accum_steps * cfg.block_size * steps_done) / dt
            entry = {
                "step": step,
                "kind": "train",
                "loss": avg_lm,
                "indexer_loss": avg_indexer,
                "phase": 1 if in_warmup else 2,
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_s": tok_per_s,
                "tokens_seen": tokens_seen,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            extra = ""
            if avg_indexer is not None:
                extra = f"  L_I {avg_indexer:.4f}  phase {1 if in_warmup else 2}"
            print(
                f"[train] step {step:5d}  loss {avg_lm:.4f}{extra}  "
                f"lr {lr:.2e}  grad {grad_norm.item():.2f}  "
                f"tok/s {tok_per_s:,.0f}"
            )
            step_t0 = time.time()
            last_log_step = step

    # save final checkpoint (small enough to be useful for downstream Stage 2 cmp)
    torch.save({"model_state": m.state_dict(), "config": asdict(mcfg)}, run_dir / "final.pt")
    msg = f"[done] saved {run_dir / 'final.pt'}  total {time.time() - t0:.1f}s"
    if cfg.early_stop_patience > 0:
        msg += f"  best_val={best_val:.4f}@step{best_step}"
        if stopped_early:
            msg += "  (early-stop)"
        else:
            msg += "  (hit max_iters cap)"
    print(msg)

    # Canonical end-of-run eval: reload best-val checkpoint, sweep val + test
    # deterministically. This is the number that goes in the paper.
    best_path = run_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state"])
        ckpt_step = int(ckpt.get("step", -1))
        print(f"[final] reloaded best.pt from step {ckpt_step}")
        ckpt_source = "best"
    else:
        print("[final] no best.pt found, using final-step weights")
        ckpt_source = "final"
        ckpt_step = cfg.max_iters

    final_entry = {
        "kind": "final_eval",
        "checkpoint": ckpt_source,
        "checkpoint_step": ckpt_step,
    }
    for split_name in ("val", "test"):
        split_data = getattr(ds, split_name, None)
        if split_data is None:
            continue
        metrics = evaluate_split_full(m, split_data, cfg, device)
        for k, v in metrics.items():
            final_entry[f"{split_name}_{k}"] = v
        line = (
            f"[final] {split_name}: loss {metrics['loss']:.4f}  "
            f"bpc {metrics['bpc']:.4f}  ppl {metrics['ppl']:.2f}  "
            f"(n_pos {metrics['n_positions']:,}, windows {metrics['n_windows']})"
        )
        if "bpc_topk" in metrics:
            line += (
                f"  ||  topk: loss {metrics['loss_topk']:.4f}  "
                f"bpc {metrics['bpc_topk']:.4f}  ppl {metrics['ppl_topk']:.2f}"
            )
        print(line)
    log_f.write(json.dumps(final_entry) + "\n")
    log_f.flush()
    log_f.close()


if __name__ == "__main__":
    train(parse_args())
