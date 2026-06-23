#!/usr/bin/env bash
# Stage E headline sweep: {1K,2K,4K,8K} × {vanilla MHA, CSA + V3.2 KL}
# 200M training tokens per cell, effective batch 32, seed 1337.
#
# Per-cell config (effective batch = batch * accum = 32):
#   ctx    batch  accum  iters  warmup  eval_int
#   1024     8      4    6000     600    1200
#   2048     4      8    3000     300     600
#   4096     2     16    1500     150     300
#   8192     1     32     750      75     150
#
# Wallclock estimate: ~5-7h total sequential.

set -euo pipefail

cd "$(dirname "$0")"

CONTEXTS=(1024 2048 4096 8192)
BATCHES=(8 4 2 1)
ACCUMS=(4 8 16 32)
ITERS=(6000 3000 1500 750)
WARMUPS=(600 300 150 75)
EVAL_INTS=(1200 600 300 150)

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
    local run_name="stage-e-${attn}-${short}"

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

echo "=== $(date '+%H:%M:%S') Stage E sweep complete ==="
