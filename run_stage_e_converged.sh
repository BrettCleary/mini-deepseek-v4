#!/usr/bin/env bash
# Stage E "converged" sweep: {1K,2K,4K,8K} × {vanilla MHA, CSA + V3.2 KL}
# Train each cell to early-stop convergence (val_loss stops improving for
# `patience` consecutive evals after LR warmup), capped at max_iters as
# safety. Best-val checkpoint saved as best.pt; final.pt is whatever was
# in memory when training stopped.
#
# Differences vs run_stage_e.sh (token-matched run):
#   - uniform max-iters=10000 across cells (was {6000,3000,1500,750})
#   - uniform eval-interval=500 (was {1200,600,300,150}); with patience=3
#     this gives a ~1500-iter no-improvement window before stopping
#   - early-stop-patience=3
#
# Cosine LR schedule decays over max_iters (10000), so cells that hit the
# cap get full cosine decay; cells that early-stop stop wherever LR happens
# to be. This is a known trade-off — for max defensibility, see the
# "post-pass" notes in the writeup.
#
# Worst-case wallclock (if no cell early-stops):
#   MHA: 1K~11min + 2K~24min + 4K~57min + 8K~2.4h = ~4h
#   CSA: 1K~23min + 2K~74min + 4K~4.2h + 8K~14.7h = ~20.5h
#   Total: ~24.5h. Early-stop typically reduces by 30-50%.

set -euo pipefail

cd "$(dirname "$0")"

CONTEXTS=(1024 2048 4096 8192)
BATCHES=(8 4 2 1)
ACCUMS=(4 8 16 32)

MAX_ITERS=10000
WARMUP_ITERS=600
EVAL_INTERVAL=500
EVAL_ITERS=50
PATIENCE=3

short_ctx() {
    case "$1" in
        1024) echo "1k" ;;
        2048) echo "2k" ;;
        4096) echo "4k" ;;
        8192) echo "8k" ;;
    esac
}

run_cell() {
    local attn=$1 ctx=$2 batch=$3 accum=$4
    local short
    short=$(short_ctx "$ctx")
    local run_name="stage-e-${attn}-${short}-converged"

    if [ -f "runs/${run_name}/final.pt" ]; then
        echo "=== $(date '+%H:%M:%S') skipping $run_name (already complete) ==="
        return
    fi

    local extra=""
    if [ "$attn" = "csa" ]; then
        extra="--csa-m 4 --csa-top-k 128 --indexer-warmup-iters $WARMUP_ITERS"
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
        --max-iters "$MAX_ITERS" \
        --eval-interval "$EVAL_INTERVAL" --eval-iters "$EVAL_ITERS" \
        --lr 3e-4 --min-lr 3e-5 --warmup-iters "$WARMUP_ITERS" \
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
            "${ACCUMS[$i]}"
    done
done

echo "=== $(date '+%H:%M:%S') Stage E converged sweep complete ==="
