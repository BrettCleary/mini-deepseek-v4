# mini-deepseek-v4

A small, locally-trainable PyTorch implementation of the **Compressed Sparse
Attention (CSA)** mechanism from DeepSeek-V4 (§2.3.1, eqs. 9–19), built to
answer one question:

> Does compressed sparse attention overtake dense attention as context length
> grows, at a scale a single GPU can reach?

The implementation is complete and tested. **The question is not yet answered** —
see [Results](#results) for what the sweeps do and don't support.

## Status

| | |
| --- | --- |
| CSA implementation (eqs. 9–19, + V3.2 auxiliary indexer KL) | complete, 24 unit tests passing |
| Context sweep on enwiki8, 1K–16K, vanilla vs CSA | run three times; **confounded**, see below |
| Crossover claim | **not supported by current data** |

## Results

All numbers are **bits per character on the held-out enwiki8 test split**,
evaluated from the best-validation checkpoint (`best.pt`), full-sequence eval.
Lower is better. Architecture is identical across arms except the attention
block: `d_model=384`, 6 layers, 6 heads (`head_dim=64`), SwiGLU `d_ff=1024`,
learned absolute positions, tied LM head, AdamW 3e-4 → 3e-5 cosine, bf16,
effective batch 32 sequences. CSA uses `m=4`, `c=64` (4× KV compression along
sequence), `top_k = n_blocks / 4` (constant 25% sparsity).

Non-embedding params: **vanilla 10.62M**, **CSA 9.86M**.

### The gap, three times

`gap = CSA − vanilla` in bpc. Three sweeps of the same grid, differing only in
iteration caps and whether early stopping fired:

| context | v2 gap | v3 gap | v3.1 gap |
| ------- | ------ | ------ | -------- |
| 1K      | 0.593  | 0.565  | —        |
| 2K      | 0.634  | 0.575  | 0.417    |
| 4K      | 0.877  | 0.542  | 0.291    |
| 8K      | 0.930  | 1.066  | 0.131    |

Three sweeps, three incompatible stories: the gap **widens** with context (v2),
is **flat then blows up** (v3), or **collapses toward crossover** (v3.1). The
architecture and data are identical in all three. Only the training schedule
changed.

### Why: the LR schedule is tied to the iteration cap

`lr_at()` in `train.py` anneals cosine over `max_iters`. Early stopping halts a
run before `max_iters`. So **a run that early-stops never finishes its LR
anneal**, and cells that stop at different fractions of their cap are not
comparable.

*(Fixed for future runs via `--lr-horizon`; see [What would fix it](#what-would-fix-it).
Every number in the tables on this page predates that fix.)*

The cleanest demonstration is a pair of runs already in this repo:

| run | cap | best step | test bpc |
| --- | --- | --- | --- |
| `stage-e-v3-vanilla-8k`   | 2500 | 2500 | **1.625** |
| `stage-e-v3-1-vanilla-8k` | 8000 | 2500 | **1.769** |

Same seed, same context, same batch/accum, same best step, same data. The only
difference is the cosine horizon, and it is worth **0.144 bpc** — *larger than
the entire 8K gap of 0.131* that v3.1 reports.

The bias is systematic, not random. In v3.1 every vanilla cell early-stopped
early in its schedule while every CSA cell ran deep into its own:

| context | vanilla: best step / cap | CSA: best step / cap |
| ------- | ------------------------ | -------------------- |
| 2K      | 6000 / 30000 (20%)       | 17200 / 30000 (57%)  |
| 4K      | 3000 / 15000 (20%)       | 7700 / 15000 (51%)   |
| 8K      | 2500 / 8000 (31%)        | 7400 / 8000 (93%)    |

CSA received a substantially fuller LR anneal than vanilla in **every** cell.
The apparent collapse of the gap is what that bias looks like.

One further caution: the trend line `0.565 → 0.417 → 0.291 → 0.131` splices the
1K cell from v3 with the 2K/4K/8K cells from v3.1. Within v3 alone the gaps are
`0.565 / 0.575 / 0.542 / 1.066` — flat, then a jump. The "collapse" tracks the
sweep it came from, not the context length.

### Raw numbers

| context | v2 van | v2 CSA | v3 van | v3 CSA | v3.1 van | v3.1 CSA |
| ------- | ------ | ------ | ------ | ------ | -------- | -------- |
| 1K  | 1.375 | 1.968 | 1.338 | 1.903 | — | — |
| 2K  | 1.420 | 2.054 | 1.352 | 1.927 | 1.419 | 1.836 |
| 4K  | 1.565 | 2.443 | 1.506 | 2.048 | 1.643 | 1.934 |
| 8K  | 2.250 | 3.180 | 1.625 | 2.691 | 1.769 | 1.900 |

The 16K probe (v3.2) is half-built and sits outside the table: **vanilla-16k
scored 1.570 bpc** (`stage-e-v3-2-vanilla-16k`, cap 10000, schedule completed),
while `stage-e-v3-2-csa-16k` died with a CUDA OOM in the `F.softmax(scores)`
path (`model.py:341`) on a 40GB A100. The most interesting context has no CSA
arm.

That vanilla number exposes a third problem: **16K (1.570) scores better than 8K
(1.625)**. Longer context should not be *easier*. This says the 8K vanilla cells
are undertrained, which means the true 8K gap is *wider* than any number in the
tables above.

### What this repo currently supports

- CSA at 4× compression is **consistently and substantially worse** than dense
  attention at every context length measured, by 0.13–1.07 bpc.
- No sweep here is a clean converged comparison. Vanilla and CSA have never been
  run to convergence under a matched LR schedule.
- **No crossover has been observed, and none of these numbers can establish or
  rule one out.**

### What would fix it

1. ~~Decouple the LR schedule from `max_iters`.~~ **Done.** `--lr-horizon` sets
   the anneal length independently of the stopping cap, and `--lr-schedule wsd`
   adds a flat-then-decay alternative. Unset, `--lr-horizon` reproduces the old
   behaviour exactly, so the runs already in `runs/` stay reproducible; the
   trainer now prints a warning when early stopping is on and no horizon is set.
2. ~~Make `csa-16k` fit in 40GB.~~ **Partly done** — see below.
3. **Re-run the sweep.** Not started, and it should be *one* sweep covering the
   whole ladder rather than new 8K/16K cells spliced onto the old short-context
   ones. The LR bias applied to every cell, not just the long ones (at 2K,
   vanilla stopped at 20% of its cap and CSA at 57%), so mixing schedules across
   contexts rebuilds the same splice this README already flags.

Item 3 is now the only thing standing between this repo and a clean answer, and
it is the expensive one: roughly 40-60 A100-hours at v3.1's measured rates
(`csa-8k` alone took 8.5h).

#### On the 16K memory work

The `run_stage_e_v3_2.sh` estimate ("768MB/layer") was wrong on two counts. The
score matrix is `(b, n_heads, n, n_blk)`, and **autocast runs `softmax` in fp32
regardless of the bf16 input** — so at 16K one layer's attention probabilities
are 1.5GB, not 768MB, and a bf16 copy is made on the way into the output
matmul. Measured saved-for-backward totals at the true 16K shape:

| | per layer | 6 layers, fwd+bwd |
| --- | --- | --- |
| before | 3.79 GiB | ~25.0 GiB |
| after  | 3.73 GiB | ~22.9 GiB |

Two exactly output-preserving changes (both covered by tests):

- **`--csa-chunk`** splits core attention along the query axis, so the transient
  mask/softmax copies scale with the chunk instead of `block_size`. This is
  where most of the saving is: at 16K the forward transient drops from 2.21 GiB
  to 0.36 GiB.
- **The indexer KL no longer materializes `log_softmax(I)`.** Using
  `KL = Σp·log p − Σp·I + logsumexp(I)·Σp` keeps only a `(b, n)` reduction
  instead of a full `(b, n, n_blk)` fp32 tensor. This is also why attention
  masks now use a large finite value instead of `-inf`: `p·I` at a masked block
  must be `0`, not `0 · -inf = NaN`.

**This is not yet confirmed to fix the OOM.** The projection above (~23 GiB) is
well under the A100's 40GB, but the run that died reported 36.9GB *allocated*,
and that gap is unexplained — it cannot be reproduced on the 16GB local card.
Confirm with a short `csa-16k` smoke run before committing to a full sweep. The
largest remaining lever, if it is still tight, is computing the core softmax in
bf16 (~9GB at 16K), but that changes numerics mid-study and should be a last
resort.

## Setup

Requires Python 3.12 and a CUDA GPU.

```bash
uv venv --python 3.12
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install -r requirements.txt
```

(The CUDA-12.8 wheel index is needed for Blackwell-class GPUs; CPU-only or
older-CUDA setups can use the default index.) Local development was on an
RTX 5070 Ti; the Stage E sweeps ran on H100 PCIe and A100 40GB instances — see
[`LAMBDA_LAUNCH.md`](LAMBDA_LAUNCH.md).

## Run

```bash
# vanilla baseline
python train.py --run-name baseline --dataset enwiki8 --block-size 1024 --max-iters 3000

# CSA
python train.py --run-name csa --dataset enwiki8 --attention csa \
    --block-size 1024 --csa-m 4 --csa-top-k 64 --max-iters 3000

# tests
python tests/test_csa.py
python tests/test_data.py

# plot
python plot.py baseline csa
```

Full sweeps are the `run_stage_e*.sh` scripts. Logs land in
`runs/<name>/log.jsonl` (one JSON object per train/eval event), checkpoints in
`runs/<name>/{best,final}.pt` (gitignored — too large).

## Configuration

`train.py --help` lists everything; flags are generated from the `TrainConfig`
dataclass. The ones that matter:

| Flag | Default | What |
| ---- | ------- | ---- |
| `--attention` | `vanilla` | `vanilla` or `csa` |
| `--dataset` | `tinyshakespeare` | `tinyshakespeare` or `enwiki8` |
| `--block-size` | 1024 | training sequence length |
| `--batch-size` | 32 | micro-batch |
| `--grad-accum-steps` | 1 | micro-batches per optimizer step |
| `--amp` | `none` | `none` or `bf16` (halves activation memory) |
| `--max-iters` | 3000 | stopping backstop |
| `--lr-horizon` | 0 | steps the LR anneal spans; 0 falls back to `--max-iters`. **Set this whenever early stopping is on** — see Results |
| `--lr-schedule` | `cosine` | `cosine` or `wsd` (flat, then linear decay over the last `--wsd-decay-frac`) |
| `--csa-chunk` | 0 | query-axis chunk for CSA attention; 0 = one shot. Memory only, results identical |
| `--early-stop-patience` | 0 | evals without val improvement before stopping; 0 disables |
| `--lr` / `--min-lr` | 3e-4 / 3e-5 | cosine endpoints |
| `--csa-m` | 4 | compression factor (m KV tokens → 1 compressed entry) |
| `--csa-top-k` | 16 | eval-time top-k blocks per query |
| `--indexer-warmup-iters` | 0 | V3.2 two-phase: indexer-only warmup steps |

## Design

Choices for this study, justified in [`notes.md`](notes.md) as decisions D1–D6:

- **No RoPE** — learned absolute positions, identical in both arms, so the
  comparison isolates the attention mechanism. Costs length extrapolation.
- **RMSNorm pre-norm**, **SwiGLU MLP**, **tied LM head**.
- **Byte-level** on enwiki8 (vocab 205); char-level on TinyShakespeare (vocab 65).
- **AdamW**, betas (0.9, 0.95), weight decay 0.1 on 2D params only, grad clip 1.0.

### Known limitations

- **First `m` positions are masked from the loss** in the CSA arm — they have no
  causally-valid compressed block (D2). Vanilla eval includes all positions. At
  4-of-1024 this is 0.4%, well below the effects of interest, and not corrected.
- **Dense-train / top-k-eval mismatch** (D3). The V3.2 auxiliary indexer KL loss
  is implemented (`model.py:464`) to address this; the sweeps above report both
  dense and top-k numbers in `log.jsonl`.
- **The LR-schedule confound described in Results is not yet fixed.**

### Out of scope

MoE, FP4/FP8, Muon, mHC residuals, HCA, sliding-window attention, attention
sink, multi-GPU, gradient checkpointing, FlashAttention. Each is its own project.

## Repo layout

```
model.py              # transformer + CSA attention, indexer, V3.2 KL loss
train.py              # training loop, --attention {vanilla,csa}
data.py               # tinyshakespeare (char) + enwiki8 (byte) loaders, 90/5/5
plot.py               # loss-curve plotting
tests/                # 24 unit tests for the CSA math
LICENSE               # MIT
notes.md              # dimension table, design decisions D1-D6, future work
run_stage_e*.sh       # context sweeps (v1, v2, v3, v3.1, v3.2)
lambda_*.sh           # cloud GPU provisioning helpers
LAMBDA_LAUNCH.md      # sweep launch checklist
runs/<name>/          # per-run config.json + log.jsonl
results/              # plots
```

`runs/` also contains the earlier TinyShakespeare Stage 1–4 build runs
(`baseline-v1`, `csa-stage{2,3,4}-v1`) that validated each construction step
against the paper's equations, and `results/` the corresponding plots. They are
superseded by the enwiki8 sweeps and are kept only as build history.

## License

MIT — see [`LICENSE`](LICENSE).
