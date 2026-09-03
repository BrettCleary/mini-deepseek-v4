"""Does more context actually help on this dataset?

Loads a trained checkpoint and reports test bits-per-character bucketed by
position within the evaluation window. Position t has t tokens of context
available, so the curve *is* the answer: if bpc keeps falling as t grows,
longer context is buying prediction quality; if it flattens, the context axis
the Stage E sweeps vary is inert past that point.

This measures one model at many context lengths, so it is free of the
cross-cell confounds (LR horizon, parameter count, positional-embedding
budget) that separate cells of a sweep.

    python position_bpc.py runs/stage-e-v3-2-vanilla-16k [--max-windows 200]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import data
import model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
    ap.add_argument("--buckets", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.run_dir / args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    # Tolerate checkpoints written before newer ModelConfig fields existed.
    known = set(model.ModelConfig.__dataclass_fields__)
    mcfg = model.ModelConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    # Checkpoints written before vanilla QK-norm existed have no q_norm/k_norm
    # weights, and their saved config predates the field, so the dataclass
    # default (True) would build an architecture the state dict cannot fill.
    # Infer it from the weights instead.
    if mcfg.attention == "vanilla" and "vanilla_qk_norm" not in cfg_dict:
        has_norm = any(".attn.q_norm." in k for k in ckpt["model_state"])
        if mcfg.vanilla_qk_norm != has_norm:
            print(f"[compat] checkpoint predates vanilla_qk_norm; using {has_norm}")
            mcfg.vanilla_qk_norm = has_norm
    m = model.MiniTransformer(mcfg).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()

    ds = data.load(json.loads((args.run_dir / "config.json").read_text())["train"]["dataset"])
    split = getattr(ds, args.split)
    L = mcfg.block_size

    doc_starts = ds.starts_for(args.split)
    if doc_starts is not None:
        # Restart the sweep at every document: a window spanning two documents
        # would put unrelated text in the "context" whose value we are measuring.
        starts = data.DocumentSampler(doc_starts, split.numel(), L).eval_windows(L)
    else:
        starts = list(range(0, split.numel() - L - 1, L))
    if args.max_windows:
        starts = starts[: args.max_windows]

    # Sum of per-position NLL (nats) across windows, kept per position.
    pos_nll = torch.zeros(L, dtype=torch.float64, device=device)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    with torch.no_grad():
        for i, s in enumerate(starts):
            x = split[s : s + L].unsqueeze(0).to(device)
            y = split[s + 1 : s + L + 1].unsqueeze(0).to(device)
            with amp:
                logits, _, _ = m(x)
            nll = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none"
            )
            pos_nll += nll.double()
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(starts)} windows", flush=True)

    bpc = (pos_nll / len(starts)) / math.log(2)      # (L,) bpc at each position

    print(f"\n{args.run_dir.name}  |  {args.split} split  |  "
          f"{len(starts)} windows x {L} positions  |  overall {bpc.mean():.4f} bpc\n")
    print(f"{'positions (context available)':>32} | {'bpc':>7} | vs prev")
    print("-" * 56)
    edges = [round(L * (k / args.buckets)) for k in range(args.buckets + 1)]
    prev = None
    for a, b in zip(edges[:-1], edges[1:]):
        v = bpc[a:b].mean().item()
        delta = "" if prev is None else f"{v - prev:+.4f}"
        print(f"{f'{a:>6}-{b:<6}':>32} | {v:7.4f} | {delta:>8}")
        prev = v

    # Headline: does the second half of the window beat the first?
    early = bpc[L // 8 : L // 4].mean().item()
    late = bpc[-L // 8 :].mean().item()
    print(f"\n  early window ({L//8}-{L//4}):  {early:.4f} bpc")
    print(f"  late  window ({L - L//8}-{L}):  {late:.4f} bpc")
    print(f"  gain from {L//4} -> {L} tokens of context: {early - late:+.4f} bpc")


if __name__ == "__main__":
    main()
