"""Status of the pg19 gate run, against the diverged first attempt.

The failure mode to watch for is gradient-norm growth: attempt 1 sat near 0.3
through step ~1000, reached 24 by step 3000 and 4.3e5 by the end, with best val
stranded at step 2100. If the QK-norm run stays flat through step ~3000 the fix
held.

    python gate_status.py [run_name] [--vs baseline_run]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rows(name: str) -> list[dict]:
    p = Path("runs") / name / "log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="pg19-gate-16k-qknorm")
    ap.add_argument("--vs", default="pg19-gate-16k")
    args = ap.parse_args()

    cur, ref = rows(args.run), rows(args.vs)
    if not cur:
        raise SystemExit(f"no log for runs/{args.run}")

    ev = [r for r in cur if r.get("kind") == "eval"]
    tr = [r for r in cur if r.get("kind") == "train"]
    ln2 = math.log(2)

    if ev:
        last = ev[-1]
        best_bpc = last.get("best_val", last["val_loss"]) / ln2
        hrs = last.get("elapsed_s", 0) / 3600
        print(f"{args.run}: step {last['step']}/10000  "
              f"val {last['val_loss']/ln2:.4f} bpc  "
              f"best {best_bpc:.4f}@{last.get('best_step')}  "
              f"{hrs:.2f}h elapsed"
              + (f"  (eta {hrs/max(last['step'],1)*10000:.1f}h)" if last["step"] else ""))
        # Divergence tell: best val stuck far behind the current step.
        if last.get("best_step") is not None and last["step"] - last["best_step"] > 2000:
            print("  !! best val is >2000 steps stale -- likely diverging again")

    refg = {r["step"]: r["grad_norm"] for r in rows(args.vs) if r.get("kind") == "train"}
    print(f"\n{'step':>6} | {'grad (this run)':>16} | {'grad ('+args.vs+')':>28}")
    print("-" * 58)
    for r in tr[::max(1, len(tr) // 20)]:
        o = refg.get(r["step"])
        print(f"{r['step']:>6} | {r['grad_norm']:>16.3f} | "
              f"{'--' if o is None else f'{o:.3f}':>28}")


if __name__ == "__main__":
    main()
