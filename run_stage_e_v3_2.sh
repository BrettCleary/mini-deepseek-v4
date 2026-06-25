#!/usr/bin/env bash
# Stage E v3.2: 16K crossover probe. Test whether dense CSA actually beats
# dense vanilla at 16K, extrapolating the v3.1 gap-collapse trend
# (0.565 → 0.417 → 0.291 → 0.131 bpc).
#
# Single cap=10000 for both architectures. Early-stop + best.pt mechanism
# already saves the min-val checkpoint, so even if vanilla becomes unstable
# past its plateau (as it did at 8K), we report the best-val number.
# Cleaner methodology than per-architecture caps.
#
# Same patience=20, eval_int=100 as v3.1 (2000-step patience window).
# csa_top_k = n_blocks/4 = 16384/4/4 = 1024.
#
# Memory at 16K, A100 40GB:
#   Vanilla: SDPA / flash-attn, O(N) memory, trivial.
#   CSA: attn scores N×n_blocks×n_heads×bf16 = 16K×4K×6×2B = 768MB/layer
#        × 6 layers ≈ 4.6GB just for attn scores. Easily fits.
#
# Wallclock est (A100): vanilla-16k ~1-2h, csa-16k ~4-7h. Total ~5-9h, ~$10-18.

set -euo pipefail

cd "$(dirname "$0")"

CTX=16384
BATCH=1
ACCUM=32
MAX_ITERS=10000
WARMUP=500
TOPK=1024
EVAL_INT=100
PATIENCE=20

run_cell() {
    local attn=$1
    local run_name="stage-e-v3-2-${attn}-16k"

    if [ -f "runs/${run_name}/final.pt" ]; then
        echo "=== $(date '+%H:%M:%S') skipping $run_name (already complete) ==="
        return
    fi

    local extra=""
    if [ "$attn" = "csa" ]; then
        extra="--csa-m 4 --csa-top-k $TOPK --indexer-warmup-iters $WARMUP"
    fi

    echo "=== $(date '+%H:%M:%S') launching $run_name ==="
    # shellcheck disable=SC2086
    python train.py \
        --attention "$attn" \
        --run-name "$run_name" \
        --dataset enwiki8 \
        --d-model 384 --n-layers 6 --n-heads 6 \
        --block-size "$CTX" \
        --batch-size "$BATCH" --grad-accum-steps "$ACCUM" \
        --max-iters "$MAX_ITERS" \
        --eval-interval "$EVAL_INT" --eval-iters 50 \
        --lr 3e-4 --min-lr 3e-5 --warmup-iters "$WARMUP" \
        --weight-decay 0.1 --grad-clip 1.0 \
        --amp bf16 --seed 1337 \
        --early-stop-patience "$PATIENCE" \
        $extra
    echo "=== $(date '+%H:%M:%S') finished $run_name ==="
}

run_cell vanilla
run_cell csa

echo "=== $(date '+%H:%M:%S') Stage E v3.2 (16K) sweep complete ==="
