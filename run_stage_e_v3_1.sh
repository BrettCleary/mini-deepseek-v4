#!/usr/bin/env bash
# Stage E v3.1: re-run 2K/4K/8K cells with much higher max-iters so early-stop
# actually fires. v3 had max-iters set to fixed-token-budget per cell, but
# convergence point grows with context, so 4K and 8K hit the cap.
#
# Skip 1K — already early-stopped cleanly in v3 (vanilla@14700, csa@18000).
#
# Per-cell cap changes vs v3:
#   ctx    v3 cap   v3.1 cap   reason
#   2048   10000    30000      v3 vanilla best@8400, csa best@9400 (just shy)
#   4096    5000    15000      v3 vanilla best@4900, csa best@5000 (at wire)
#   8192    2500     8000      v3 both best at last eval (val still descending)
#
# Wallclock est on H100 PCIe: ~12-15h if early-stop fires at ~60% of cap on
# average (~$30-37). Worst case all hit cap: ~22h, $55.
#
# Same patience=20, eval_interval=100, top_k=n_blocks/4 as v3. Only max-iters
# and warmup-iters change.

set -euo pipefail

cd "$(dirname "$0")"

CONTEXTS=(2048 4096 8192)
BATCHES=(4 2 1)
ACCUMS=(8 16 32)
ITERS=(30000 15000 8000)
WARMUPS=(1500 750 400)
TOPKS=(128 256 512)
EVAL_INT=100
PATIENCE=20

short_ctx() {
    case "$1" in
        2048) echo "2k" ;;
        4096) echo "4k" ;;
        8192) echo "8k" ;;
    esac
}

run_cell() {
    local attn=$1 ctx=$2 batch=$3 accum=$4 iters=$5 warmup=$6 topk=$7
    local short
    short=$(short_ctx "$ctx")
    local run_name="stage-e-v3-1-${attn}-${short}"

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

echo "=== $(date '+%H:%M:%S') Stage E v3.1 sweep complete ==="
