"""Where does long context actually pay off?

Average bpc answers "how much does context help the typical byte?" -- the wrong
question if the payoff is concentrated in a small subset of predictions. A
handful of tokens gaining several bits each, diluted across 16K positions, is
worth ~0.02 bpc on average and is invisible in the mean.

Method. Each token is scored twice on sequences that are identical in length,
in absolute positions, and in local context -- differing only in whether the
*distant* context carries real information:

  clean     -- the true window
  corrupted -- bytes [0, --corrupt-before) replaced by text from another book

The difference is the number of bits the genuine distant context is worth for
that specific byte.

Two earlier designs were wrong and are worth recording. Masking attention to a
short window at eval takes a model trained with full attention off its training
distribution. Re-feeding just the recent bytes as their own sequence is worse
still: the sequence then has no position-0 token, and transformers lean on an
early-token attention sink, so the model collapses to near-random (5.7 bpc
against 6.98 for a uniform guess over this vocab). Both produced "gains" of
4+ bpc, larger than the model's entire bpc -- a measurement artefact, not an
effect. Swapping in real text from another document keeps length, positions and
the sink intact, so only information content changes.

    python context_value.py runs/pg19-gate-16k-qknorm --corrupt-before 8192
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import data
import model


def logits_at(m, idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Forward pass with explicit absolute position indices."""
    x = m.tok_embed(idx) + m.pos_embed(pos)
    for blk in m.blocks:
        x, _ = blk(x)
    return m.lm_head(m.norm_f(x))


def nll(m, idx, pos, targets):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = logits_at(m, idx, pos)
    return F.cross_entropy(
        lg.float().view(-1, lg.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--corrupt-before", type=int, default=8192,
                    help="bytes before this offset are replaced with other text")
    ap.add_argument("--score-from", type=int, default=12288,
                    help="only score positions at or beyond this offset, so every "
                         "scored token keeps a margin of genuine local context")
    ap.add_argument("--max-windows", type=int, default=60)
    args = ap.parse_args()
    P, S = args.corrupt_before, args.score_from

    ckpt = torch.load(args.run_dir / "best.pt", map_location="cuda", weights_only=False)
    cfgd = ckpt["config"]
    known = set(model.ModelConfig.__dataclass_fields__)
    mcfg = model.ModelConfig(**{k: v for k, v in cfgd.items() if k in known})
    if mcfg.attention == "vanilla" and "vanilla_qk_norm" not in cfgd:
        mcfg.vanilla_qk_norm = any(".attn.q_norm." in k for k in ckpt["model_state"])
    m = model.MiniTransformer(mcfg).cuda()
    m.load_state_dict(ckpt["model_state"])
    m.eval()

    dsname = json.loads((args.run_dir / "config.json").read_text())["train"]["dataset"]
    ds = data.load(dsname)
    split, L = ds.test, mcfg.block_size
    st = ds.starts_for("test")
    starts = (
        data.DocumentSampler(st, split.numel(), L).eval_windows(L)
        if st is not None else list(range(0, split.numel() - L - 1, L))
    )[: args.max_windows]

    pos = torch.arange(L, device="cuda").unsqueeze(0)
    gains, cleans = [], []
    for i, s in enumerate(starts):
        seq = split[s : s + L + 1].cuda()
        idx, tgt = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
        # Donor text for the corrupted prefix: a different window, so it is real
        # prose from another book rather than noise.
        d = starts[(i + len(starts) // 2) % len(starts)]
        donor = split[d : d + P].cuda()
        corrupt = idx.clone()
        corrupt[0, :P] = donor

        clean_nll = nll(m, idx, pos, tgt)[0]
        corr_nll = nll(m, corrupt, pos, tgt)[0]
        gains.append((corr_nll[S:] - clean_nll[S:]).float().cpu())
        cleans.append(clean_nll[S:].float().cpu())
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(starts)}", flush=True)

    g = torch.cat(gains) / math.log(2)
    base = torch.cat(cleans) / math.log(2)
    n = g.numel()
    total = g.sum().item()

    print(f"\n{args.run_dir.name}: real vs foreign context before byte {P}, "
          f"scoring positions {S}+")
    print(f"{n:,} scored positions over {len(starts)} windows")
    print(f"  bpc, true distant context   : {base.mean():.4f}")
    print(f"  bpc, foreign distant context: {(base + g).mean():.4f}")
    print(f"  mean gain             : {g.mean():+.4f} bpc   <- the averaged number\n")
    print(f"  median gain           : {g.median():+.4f} bpc")
    print(f"  tokens helped         : {(g > 0).float().mean()*100:.1f}%   "
          f"hurt: {(g < 0).float().mean()*100:.1f}%")
    print()
    for thr in (0.5, 1.0, 2.0, 4.0):
        frac = (g > thr).float().mean().item() * 100
        share = g[g > thr].sum().item() / total * 100 if total else 0
        print(f"  gaining >{thr:>4} bits : {frac:6.3f}% of tokens, "
              f"{share:6.1f}% of all bits gained")
    for pct in (1, 5, 10):
        k = max(1, n * pct // 100)
        top = g.topk(k).values
        print(f"  top {pct:>2}% of tokens : {top.mean():+.3f} bpc mean, "
              f"{top.sum().item()/total*100:5.1f}% of all bits gained" if total else "")


if __name__ == "__main__":
    main()
