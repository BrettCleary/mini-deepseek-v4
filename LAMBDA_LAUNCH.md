# Lambda Cloud sweep — launch checklist

Stage E v3 phase 1 (1K–8K × {vanilla, CSA}) on a single H100 PCIe.
Est wallclock 4–6h, cost ~$10–15 at $2.49/hr.

## Prerequisites
- Lambda Cloud account with payment method.
- SSH public key uploaded to your Lambda account (Settings → SSH Keys).
- Your matching private key in `~/.ssh/` so `ssh ubuntu@<ip>` just works.

## Provision

1. Lambda Cloud dashboard → **Launch Instance**.
2. Region: cheapest available with **1× H100 PCIe** (us-east, us-west, etc).
3. Filesystem: none (the data is small, no point paying for persistent storage).
4. Launch. Wait ~2 min for boot, copy the public IP.

```bash
export LAMBDA=ubuntu@<paste-ip-here>
```

## Push code + data, bootstrap

From this repo on your local box:

```bash
# Send code (excludes runs/, .git/, __pycache__) + cached enwik8 (132M).
./lambda_sync.sh push "$LAMBDA"

# SSH in and run the bootstrap.
ssh "$LAMBDA"
cd ~/mini-deepseek-v4
./lambda_bootstrap.sh
```

The bootstrap:
- prints `nvidia-smi` (confirm H100 with ~80GB),
- installs `requirements.txt`,
- runs a 50-step tinyshakespeare smoke test (~30s; confirms train.py + the new full-sweep eval work end-to-end),
- launches `run_stage_e_v3.sh` nohup'd in the background,
- prints the pid and first log lines, then returns.

Exit the SSH session (Ctrl-D). The sweep continues regardless.

## Monitor (optional — sweep runs without you)

```bash
# Tail the live sweep log from local.
./lambda_sync.sh log "$LAMBDA"

# Or pull incremental results to local at any time.
./lambda_sync.sh pull "$LAMBDA"
```

Cells in order: `vanilla-1k → csa-1k → vanilla-2k → csa-2k → vanilla-4k → csa-4k → vanilla-8k → csa-8k`. The script skips any cell whose `final.pt` already exists, so a crash mid-sweep is recoverable.

## When the sweep finishes

The sweep log ends with `Stage E v3 phase-1 sweep complete`. To confirm and pull:

```bash
ssh "$LAMBDA" 'tail -3 ~/mini-deepseek-v4/runs/stage-e-v3-sweep.log'
./lambda_sync.sh pull "$LAMBDA"
```

After `pull`, local `runs/stage-e-v3-*/` has the full results (best.pt, log.jsonl, config.json) and `runs/stage-e-v3-sweep.log` has the chronological stdout.

## TERMINATE THE INSTANCE

**Critical** — Lambda bills by the second until you terminate. From the dashboard, click the instance → **Terminate**. Confirm the instance no longer appears in the running list.

## Verify the cell count made the methodology jump

After pulling, this one-liner confirms whether early-stop fired (the v2 sweep all hit max-iters):

```bash
python3 -c "
import json
for r in ['stage-e-v3-vanilla-1k','stage-e-v3-csa-1k','stage-e-v3-vanilla-2k','stage-e-v3-csa-2k','stage-e-v3-vanilla-4k','stage-e-v3-csa-4k','stage-e-v3-vanilla-8k','stage-e-v3-csa-8k']:
    rows = [json.loads(l) for l in open(f'runs/{r}/log.jsonl')]
    cfg = json.load(open(f'runs/{r}/config.json'))
    early = any(x.get('kind')=='early_stop' for x in rows)
    best = min((x for x in rows if x.get('kind')=='eval'), key=lambda x: x['val_loss'])
    last = max(x['step'] for x in rows if x.get('kind')=='eval')
    print(f'{r:30s} best@{best[\"step\"]:5d}  last@{last:5d}  cap={cfg[\"train\"][\"max_iters\"]:5d}  {\"EARLY-STOP\" if early else \"hit-cap (BAD)\"}')
"
```

If everything reads `EARLY-STOP`, the v3 numbers are the real ones and ready for the paper. If anything reads `hit-cap`, we still didn't converge that cell and the cap or patience needs another bump.
