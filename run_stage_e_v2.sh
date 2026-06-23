#!/usr/bin/env bash
# Stage E v2: real numbers with early stopping + test-split eval.
# Replaces run_stage_e.sh, whose runs all stopped on the val descent curve.
#
# Differences vs v1:
#   - max-iters raised 2x so early stopping (not the budget) decides when to stop.
#   - eval-interval tightened to give ~30 evals per run.
#   - --early-stop-patience 6 (stop if val hasn't improved in 6 consecutive evals
#     past warmup; saves best.pt at the val-min step).
#   - End-of-run reloads best.pt and reports test bpc on a deterministic
#     full sweep of the enwiki8 test split (the canonical metric).
#
# Per-cell config (effective batch = batch * accum = 32):
#   ctx    batch  accum  max-iters  warmup  eval-int  evals
#   1024     8      4    12000       600    400        30
#   2048     4      8     6000       300    200        30
#   4096     2     16     3000       150    100        30
#   8192     1     32     1500        75     50        30
#
# Wallclock: ~10-14h sequential worst case; will be less when early-stop triggers.

set -euo pipefail

cd "$(dirname "$0")"

CONTEXTS=(1024 2048 4096 8192)
BATCHES=(8 4 2 1)
ACCUMS=(4 8 16 32)
ITERS=(12000 6000 3000 1500)
WARMUPS=(600 300 150 75)
EVAL_INTS=(400 200 100 50)
PATIENCE=6

short_ctx() {
    case "$1" in
        1024) echo "1k" ;;
        2048) echo "2k" ;;
        4096) echo "4k" ;;
        8192) echo "8k" ;;
    esac
}

run_cell() {
    local attn=$1 ctx=$2 batch=$3 accum=$4 iters=$5 warmup=$6 eval_int=$7
    local short
    short=$(short_ctx "$ctx")
    local run_name="stage-e-v2-${attn}-${short}"

    if [ -f "runs/${run_name}/final.pt" ]; then
        echo "=== $(date '+%H:%M:%S') skipping $run_name (already complete) ==="
        return
    fi

    local extra=""
    if [ "$attn" = "csa" ]; then
        extra="--csa-m 4 --csa-top-k 128 --indexer-warmup-iters $warmup"
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
        --eval-interval "$eval_int" --eval-iters 50 \
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
            "${EVAL_INTS[$i]}"
    done
done

echo "=== $(date '+%H:%M:%S') Stage E v2 sweep complete ==="
