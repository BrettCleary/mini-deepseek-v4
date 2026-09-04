"""Which arm actually runs out of road first?

CSA's claim is efficiency: 1/m the KV entries, top-k of those attended. That is
a FLOP and KV-cache argument, and it assumes the dense baseline is the one that
becomes infeasible at long context.

For a from-scratch PyTorch implementation the assumption fails, because the
comparison is not FLOPs against FLOPs. Dense attention goes through
F.scaled_dot_product_attention, which dispatches to a flash kernel: O(N) memory,
never materializing the N x N score matrix. This CSA materializes
(b, n_heads, n, n_blk) scores, and autocast runs softmax in fp32, so one 16K
layer holds 1.5GB of attention probabilities. The 4x saving in attention FLOPs
is swamped by the gap in kernel quality.

Reproducing CSA's efficiency needs the fused kernels the paper describes
building (TileLang, FP8/FP4 storage) -- which is a systems project, not an
architecture one.

    python feasibility.py
"""
from __future__ import annotations

import argparse
import time

import torch

import model


def bench(attn: str, n: int, chunk: int, layers: int, d_model: int, heads: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kw = (
        dict(csa_m=4, csa_top_k=max(1, n // 4 // 4), csa_chunk=chunk)
        if attn == "csa" else {}
    )
    try:
        cfg = model.ModelConfig(
            attention=attn, vocab_size=126, d_model=d_model, n_layers=layers,
            n_heads=heads, block_size=n, **kw,
        )
        m = model.MiniTransformer(cfg).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randint(0, 126, (1, n), device="cuda")
        y = torch.randint(0, 126, (1, n), device="cuda")
        for _ in range(2):                       # warm up kernels/allocator
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = m(x, y)
            loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = m(x, y)
            loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        tps = 3 * n / (time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**30
        del m, opt, x, y
        return f"{peak:6.2f} GiB {tps:>9,.0f} tok/s"
    except torch.OutOfMemoryError:
        return "            OOM"
    finally:
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[8192, 16384, 32768, 65536])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    args = ap.parse_args()

    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"{torch.cuda.get_device_name(0)}  ({free:.1f} GiB free)")
    print(f"batch 1, fwd+bwd+step, bf16 autocast, csa_chunk={args.chunk}\n")
    print(f"{'ctx':>7} | {'vanilla (flash SDPA)':>26} | {'CSA (m=4, 25% top-k)':>26}")
    print("-" * 68)
    for n in args.contexts:
        v = bench("vanilla", n, args.chunk, args.layers, args.d_model, args.heads)
        c = bench("csa", n, args.chunk, args.layers, args.d_model, args.heads)
        print(f"{n:>7} | {v:>26} | {c:>26}")


if __name__ == "__main__":
    main()
