#!/usr/bin/env bash
# Local-side sync helper for Lambda Cloud.
#
# Usage:
#   ./lambda_sync.sh push   ubuntu@<remote-ip>   # send code + cached data
#   ./lambda_sync.sh pull   ubuntu@<remote-ip>   # fetch runs/ back
#   ./lambda_sync.sh log    ubuntu@<remote-ip>   # tail the live sweep log
#
# Assumes SSH key is already added (lambda spins instances with your uploaded key).

set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-}"
host="${2:-}"

if [ -z "$cmd" ] || [ -z "$host" ]; then
    echo "usage: $0 {push|pull|log} ubuntu@<remote-ip>" >&2
    exit 2
fi

REMOTE_DIR="~/mini-deepseek-v4"

case "$cmd" in
    push)
        # Push code + cached enwiki8 (skip the download on the remote).
        # Exclude local-only runs and pycache to keep payload small.
        rsync -avz --delete \
            --exclude '__pycache__/' \
            --exclude '.git/' \
            --exclude '*.pyc' \
            --exclude 'runs/' \
            --exclude '.venv/' \
            ./ "$host:$REMOTE_DIR/"
        # Cached enwik8 (132M, saves ~30s of redownload). Lives under data/.
        if [ -f data/enwik8 ]; then
            ssh "$host" "mkdir -p $REMOTE_DIR/data"
            rsync -avz data/enwik8 data/enwik8.zip "$host:$REMOTE_DIR/data/"
        fi
        echo "[push] done."
        ;;
    pull)
        mkdir -p runs
        # Pull only the v3 sweep dirs + sweep log. Leaves local v1/v2 untouched.
        rsync -avz "$host:$REMOTE_DIR/runs/stage-e-v3-*" runs/
        rsync -avz "$host:$REMOTE_DIR/runs/stage-e-v3-sweep.log" runs/ || true
        echo "[pull] done. runs/stage-e-v3-*/ updated."
        ;;
    log)
        ssh "$host" "tail -f $REMOTE_DIR/runs/stage-e-v3-sweep.log"
        ;;
    *)
        echo "unknown subcommand: $cmd" >&2
        echo "usage: $0 {push|pull|log} ubuntu@<remote-ip>" >&2
        exit 2
        ;;
esac
