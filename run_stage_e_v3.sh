#!/usr/bin/env bash
# Stage E v3: real convergence numbers. Phase 1 = {1K, 2K, 4K, 8K} only.
# Phase 2 (16K, 32K) follows once methodology is confirmed.
#
# Differences vs v2:
#   - max-iters raised ~3-5x so the cap is a backstop, not a budget. Each cell
#     should early-stop well before the cap.
#   - eval-interval = 100 steps regardless of ctx, for consistent patience
#     accounting. v2's eval intervals (400/200/100/50) varied per cell, which
#     interacted badly with patience.
#   - --early-stop-patience 20 (was 6). v2's val noise repeatedly reset the
#     patience counter, so it never fired. 20 evals @ interval 100 = 2000 steps
#     of no-improvement before stopping. Forgiving but not infinite.
#   - csa_top_k = n_blocks / 4 per cell (constant 25% sparsity), instead of
#     constant top_k=128. Holds *relative* sparsity fixed across the sweep so
#     the topk-degradation effect at long context isn't a methodology artifact.
#       1K → n_blocks=256, top_k=64
#       2K → n_blocks=512, top_k=128
#       4K → n_blocks=1024, top_k=256
#       8K → n_blocks=2048, top_k=512
#   - new run-name prefix stage-e-v3- so it doesn't collide with v2.
#
# Per-cell config (effective batch = batch * accum = 32):
#   ctx    batch  accum  max-iters  warmup  eval-int  top_k
#   1024     8      4    20000      1000    100        64
#   2048     4      8    10000       500    100       128
#   4096     2     16     5000       250    100       256
#   8192     1     32     2500       125    100       512
#
# Wallclock on H100 PCIe (est): ~4-6h total for phase 1.

set -euo pipefail

cd "$(dirname "$0")"

CONTEXTS=(1024 2048 4096 8192)
BATCHES=(8 4 2 1)
ACCUMS=(4 8 16 32)
ITERS=(20000 10000 5000 2500)
WARMUPS=(1000 500 250 125)
TOPKS=(64 128 256 512)
EVAL_INT=100
PATIENCE=20

short_ctx() {
    case "$1" in
        1024) echo "1k" ;;
        2048) echo "2k" ;;
        4096) echo "4k" ;;
        8192) echo "8k" ;;
    esac
}

run_cell() {
    local attn=$1 ctx=$2 batch=$3 accum=$4 iters=$5 warmup=$6 topk=$7
    local short
    short=$(short_ctx "$ctx")
    local run_name="stage-e-v3-${attn}-${short}"

    if [ -f "runs/${run_name}/final.pt" ]; then
        echo "=== $(date '+%H:%M:%S') skipping $run_name (already complete) ==="
        return
    fi

    local extra=""
    if [ "$attn" = "csa" ]; then
        extra="--csa-m 4 --csa-top-k $topk --indexer-warmup-iters $warmup"
    fi

    echo "=== $(date '+%H:%M:%S') launching $run_name ==="
    # shellcheck disable=SC2086
    python train.py \
        --attention "$attn" \
        --run-name "$run_name" \
        --dataset enwiki8 \
        --d-model 384 --n-layers 6 --n-heads 6 \
        --block-size "$ctx" \
        --batch-size "$batch" --grad-accum-steps "$accum" \
        --max-iters "$iters" \
        --eval-interval "$EVAL_INT" --eval-iters 50 \
        --lr 3e-4 --min-lr 3e-5 --warmup-iters "$warmup" \
        --weight-decay 0.1 --grad-clip 1.0 \
        --amp bf16 --seed 1337 \
        --early-stop-patience "$PATIENCE" \
        $extra
    echo "=== $(date '+%H:%M:%S') finished $run_name ==="
}

for i in "${!CONTEXTS[@]}"; do
    for attn in vanilla csa; do
        run_cell "$attn" \
            "${CONTEXTS[$i]}" \
            "${BATCHES[$i]}" \
            "${ACCUMS[$i]}" \
            "${ITERS[$i]}" \
            "${WARMUPS[$i]}" \
            "${TOPKS[$i]}"
    done
done

echo "=== $(date '+%H:%M:%S') Stage E v3 phase-1 sweep complete ==="
