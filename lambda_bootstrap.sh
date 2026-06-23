#!/usr/bin/env bash
# Runs on a fresh Lambda Cloud instance (Ubuntu 22.04 + CUDA image).
# Installs deps, smoke-tests, kicks off Stage E v3 sweep.
#
# Usage on the remote box:
#   cd ~/mini-deepseek-v4
#   ./lambda_bootstrap.sh
#
# After this returns, the sweep is running nohup'd. tail -f sweep.log to watch.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "=== pip install -r requirements.txt ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

echo
echo "=== sanity-check torch sees CUDA ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'torch {torch.__version__}')
print(f'cuda  {torch.version.cuda}')
print(f'gpu   {torch.cuda.get_device_name(0)}')
print(f'bf16  {torch.cuda.is_bf16_supported()}')
"

echo
echo "=== smoke test: 50-iter tinyshakespeare on the new eval path ==="
mkdir -p runs
rm -rf runs/smoke-bootstrap
python3 train.py --attention vanilla --dataset tinyshakespeare \
    --max-iters 50 --eval-interval 25 --eval-iters 5 \
    --block-size 128 --batch-size 8 --warmup-iters 5 \
    --run-name smoke-bootstrap
echo "[smoke] final_eval row:"
tail -1 runs/smoke-bootstrap/log.jsonl | python3 -m json.tool
rm -rf runs/smoke-bootstrap

echo
echo "=== launching real sweep in background ==="
nohup ./run_stage_e_v3.sh > runs/stage-e-v3-sweep.log 2>&1 &
echo "sweep pid: $!"
echo "watch with: tail -f runs/stage-e-v3-sweep.log"
sleep 3
echo
echo "=== first 10 lines of sweep log ==="
head -10 runs/stage-e-v3-sweep.log || echo "(log not flushed yet)"
