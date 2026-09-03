#!/usr/bin/env bash
# PG-19 axis gate.
#
# Question: does test bpc keep improving with available context on pg19, or is
# it flat past ~2K as it is on enwiki8? Until this passes there is no point
# running a CSA-vs-dense sweep over context length, because the independent
# variable does not move the outcome. This is the control Stage E never had.
#
# Configured to match runs/stage-e-v3-2-vanilla-16k EXACTLY -- same d_model,
# layers, heads, context, batch, grad-accum, warmup, LR endpoints, seed and
# iteration count -- so the only difference between the two position curves is
# the dataset.
#
# One deviation: no --early-stop-patience. The enwiki8 run had patience 20 but
# it never fired (best step 9900 of 10000), so it completed its anneal; running
# without patience reproduces that behaviour rather than risking a truncated
# anneal here, which is the confound documented in the README.
#
# Measured on an RTX 5070 Ti: 199K tok/s at 16K, 2.20 GiB peak (of ~13.6 GiB
# free). 10000 iters x 32 x 16384 = 5.24B tokens ~= 7.3h, plus ~0.4h of eval.
#
# best.pt is written on every val improvement, so killing this early still
# leaves a usable checkpoint -- position_bpc.py reads best.pt.

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
RUN=pg19-gate-16k

$PY train.py \
    --attention vanilla \
    --run-name "$RUN" \
    --dataset pg19 \
    --d-model 384 --n-layers 6 --n-heads 6 \
    --block-size 16384 \
    --batch-size 1 --grad-accum-steps 32 \
    --max-iters 10000 \
    --lr-horizon 10000 \
    --eval-interval 100 --eval-iters 50 \
    --lr 3e-4 --min-lr 3e-5 --warmup-iters 500 \
    --weight-decay 0.1 --grad-clip 1.0 \
    --amp bf16 --seed 1337

echo "=== $(date '+%F %T') training done; measuring the context curve ==="
$PY position_bpc.py "runs/$RUN" --buckets 16 | tee "runs/$RUN/position_bpc.txt"

echo
echo "=== enwiki8 reference (stage-e-v3-2-vanilla-16k), same model and budget ==="
echo "  0-1024 1.683 | 1024-2048 1.561 | 2048-3072 1.581 | 8192-9216 1.569 | 15360-16384 1.541"
echo "  i.e. flat past ~2K; 4K->16K of context was worth only ~0.04 bpc."
