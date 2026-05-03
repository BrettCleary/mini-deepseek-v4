"""Plot loss curves from one or more training runs.

Usage:
    python plot.py baseline-v1
    python plot.py baseline-v1 csa-v1               # compare side by side
    python plot.py baseline-v1 csa-v1 --output cmp.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS_DIR = Path(__file__).parent / "runs"


def read_log(run_name: str) -> tuple[list[dict], list[dict]]:
    log_path = RUNS_DIR / run_name / "log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"no log at {log_path}")
    train_entries: list[dict] = []
    eval_entries: list[dict] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        (eval_entries if e["kind"] == "eval" else train_entries).append(e)
    return train_entries, eval_entries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="one or more run names under runs/")
    p.add_argument("--output", default=None, help="output png path (default: results/<runs joined>.png)")
    p.add_argument("--ylim", type=float, nargs=2, default=None)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in args.runs:
        train_entries, eval_entries = read_log(run)
        # train: plot loss vs step (light)
        if train_entries:
            xs = [e["step"] for e in train_entries]
            ys = [e["loss"] for e in train_entries]
            ax.plot(xs, ys, alpha=0.3, label=f"{run} (train)")
        # eval: bold lines for val
        if eval_entries:
            xs = [e["step"] for e in eval_entries]
            ys = [e["val_loss"] for e in eval_entries]
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"{run} (val)")

    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if args.ylim:
        ax.set_ylim(*args.ylim)

    out = args.output
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        out = str(results_dir / ("_vs_".join(args.runs) + ".png"))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
