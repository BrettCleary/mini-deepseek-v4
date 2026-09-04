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

### The bigger problem: the CSA arm has a recency hole

Paper §2.3.3 is explicit that the compressed branch alone leaves a blind spot:

> "In order to strictly preserve causality in CSA and HCA, each query attends to
> only preceding compressed KV blocks. **Consequently, a query cannot access
> information from other tokens within its own compressed block.** ... For these
> reasons, we introduce a supplementary attention branch [in] a sliding window
> manner."

Figure 3 — the CSA architecture diagram — shows those sliding-window entries
concatenated with the selected compressed entries before Shared-KV MQA.
Equations 9–19 render only the compressed branch, so this implementation, built
faithfully from the equations, omitted it. A gradient probe confirms the gap:
at `m=4`, `logits[23]` has *exactly zero* dependence on input positions 20, 21,
22, at any depth. Every query is missing the `t mod m` bytes immediately before
it — ~1.5 bytes on average, and at byte level those are the most informative
ones. An order-K n-gram proxy prices the loss at **~+0.52 bpc**, comparable to
the entire gap the tables above report.

Implemented as `--csa-n-win` (§2.3.3; DeepSeek-V4 uses `n_win = 128` at `m = 4`,
§4.2.1). See notes.md D7 for the design choices where the paper is ambiguous.

### The largest confound: only one arm had QK-norm

`CSAAttention` has applied RMSNorm to queries and compressed KV entries since
Stage 2 — paper §2.3.3, whose stated purpose is that it "avoids exploding
attention logits and may improve training stability".
`VanillaMultiHeadAttention` had none; it went straight from the qkv projection
into SDPA. D4 was careful to give both arms the same positional encoding, but
normalization was never equalized.

With `--grad-clip 1.0`, and `grad_norm` logged pre-clip:

| arm | ctx | median grad norm | steps clipped | effective LR |
| --- | --- | --- | --- | --- |
| vanilla | 1K | 0.52 | 10.8% | ×1.00 |
| vanilla | 2K | 0.77 | 41.5% | ×1.00 |
| vanilla | 8K | 10.69 | **91.0%** | **×0.094** |
| vanilla | 16K | 15.42 | **94.0%** | **×0.065** |
| CSA | 1K | 0.58 | 0.3% | ×1.00 |
| CSA | 2K | 0.52 | 0.8% | ×1.00 |
| CSA | 8K | 0.62 | 17.6% | ×1.00 |

Vanilla's gradient norms grow with context; CSA's do not. At 8K and 16K the
baseline was taking steps roughly a tenth of their nominal size while CSA ran
essentially unclipped. This is arm-asymmetric *and* context-dependent, biting
hardest exactly where the study's claim lives, and it plausibly accounts for
vanilla early-stopping at 20-31% of its cap at long context, for vanilla bpc
worsening with context in all three sweeps, and for much of v3.1's apparent
gap collapse at 8K.

It surfaced when the pg19 gate run diverged outright (gradient norms reaching
4.3e5, best val at step 2100 of 10000, then ~50% degradation). Fixed by adding
QK-RMSNorm to the dense baseline — `--vanilla-qk-norm`, on by default, with
`--no-vanilla-qk-norm` to reproduce earlier runs. **Every arm comparison in the
tables on this page predates the fix.**

**The three known confounds do not share a direction**, which is why those
tables cannot be corrected by inspection:

| confound | flatters | reported gap is therefore |
| --- | --- | --- |
| LR anneal tied to the cap (CSA got 51-93% of its schedule, vanilla 20-31%) | CSA | too **narrow** |
| QK-norm on CSA only (vanilla at ~1/10 effective LR at 8K-16K) | CSA | too **narrow** |
| recency hole (~0.5 bpc) | vanilla | too **wide** |

### The context axis is nearly inert on this dataset

Before trusting any sweep *over context length*, it is worth asking whether
context length does anything here at all. It mostly doesn't.

`position_bpc.py` measures one trained model's test bpc bucketed by position
within the eval window — position `t` has `t` tokens of context available, so
the curve is the answer, and using a single model avoids every cross-cell
confound (LR horizon, parameter count, positional-embedding budget).

`stage-e-v3-2-vanilla-16k` is the right model to ask: it reached its best step
at 9900 of a 10000 cap, so it essentially completed its anneal and is the one
run the LR confound does not touch. Full test split, 305 windows:

| positions (context available) | bpc |
| --- | --- |
| 0-1024 | 1.683 |
| 1024-2048 | 1.561 |
| 2048-3072 | 1.581 |
| 4096-5120 | 1.580 |
| 8192-9216 | 1.569 |
| 12288-13312 | 1.549 |
| 15360-16384 | 1.541 |

One drop as the model acquires its first ~2K of context, then flat. **Going from
4K to 16K of context is worth ~0.04 bpc**, against bucket-to-bucket noise of
±0.02-0.04. At 1K the same measurement (`stage-e-v3-vanilla-1k`, 64-position
buckets) shows most of the gain arriving in the first ~256 positions: 1.683 at
0-64, 1.301 by 192-256, then broadly flat to 1.24 at 1024.

**This is the finding that undercuts the Stage E design.** The sweep varies an
independent variable worth ~0.04 bpc while trying to detect architecture
differences of 0.13-1.07 bpc — the effect sought is an order of magnitude
larger than the total headroom on the axis being varied. It also explains why
vanilla bpc got *worse* with context in all three sweeps: longer context buys
almost nothing while costing real capacity (16384 learned position embeddings,
each receiving far fewer updates) and optimizer steps.

Caveats, because this is a claim about a regime and not about enwik8 in
general: these are ~10M-parameter models at 1.3-1.6 bpc, well short of tuned
char-level models (~1.0), so capacity rather than data may be the binding
constraint; and D4's learned absolute position embeddings (no RoPE) plausibly
cap long-range use on their own. The honest statement is that *at this scale,
with these positional embeddings*, context beyond ~2K contributes almost
nothing — not that enwik8 lacks long-range structure.

### PG-19, and the gate that has to come first

`--dataset pg19` adds Project Gutenberg books (Rae et al. 2019), the standard
long-range LM benchmark, at the same 90M / 5M / 5M byte-level split sizes as
enwiki8 so bpc is directly comparable. Books are fetched per-title from the
public `deepmind-gutenberg` GCS bucket (no `datasets` dependency), taking a
seeded *random* subset rather than the lowest Gutenberg ids — those correlate
with age and canon status, and the first book by id is the 4.4MB King James
Bible.

The structural argument for it is solid: PG-19 books median ~400KB, so a
window stays inside one book even at 64K. Windows are now document-aware
everywhere (`data.DocumentSampler`) — training batches, periodic eval, and the
full-split sweeps all restart at document boundaries, and documents shorter
than `block_size` are skipped. On enwiki8 that mattered for 41% of bytes at
16K context.

The argument *against* it is unresolved: PG-19's long-range signal is largely
semantic (plot, characters, style), and a ~10M-parameter byte-level model
mostly cannot use that. Code would offer more mechanically exploitable
long-range structure — repeated identifiers, imports, call sites — which is
what induction heads actually learn.

**So PG-19 is not yet justified, and no sweep should be run on it until it
passes the gate:** train one vanilla model at long context and check
`position_bpc.py` shows bpc still falling well past 2K. That is one cheap run,
and it is the control that Stage E never had. If the curve is flat again, the
dataset is not the fix and code is the next candidate.

`run_pg19_gate.sh` runs exactly that. It is configured to match
`stage-e-v3-2-vanilla-16k` in every respect — model, context, batch,
grad-accum, warmup, LR endpoints, seed, iteration count — so the only
difference between the two position curves is the dataset. Measured on an
RTX 5070 Ti: 199K tok/s at 16K, 2.20 GiB peak, ~7.6h for the full 10000 iters.
`best.pt` is written on every val improvement, so an interrupted run still
yields a usable curve.

Progress is in `runs/pg19-gate-16k/log.jsonl` (the `.log` file stays empty
until the process exits, since Python buffers stdout when it is not a TTY).

**Result (2026-09-04): pg19 does not rescue the design.** With QK-norm and the
vocab fix, the run was healthy — gradient norms flat at 0.33-0.50 for all 10000
steps, best val 1.4918 bpc at step 9700, anneal completed. So this is a
well-trained model, and the earlier "maybe it was just undertrained" escape
hatch is closed.

| positions (context available) | bpc |
| --- | --- |
| 0-1024 | 1.654 |
| 1024-2048 | 1.546 |
| 2048-3072 | 1.549 |
| 4096-5120 | 1.524 |
| 8192-9216 | 1.517 |
| 12288-13312 | 1.512 |
| 15360-16384 | 1.513 |

One drop over the first ~2K, then a slow drift. A paired per-window test (early
slice 2-4K of context vs late slice 14-16K, pairing removes between-window
difficulty variation) gives **+0.0240 ± 0.0086 bpc, t = 2.8**, with 60% of
windows improving.

So the *average* gain is 0.024 bpc for an 8x increase in context. **That average
turns out to be the wrong statistic**, and the conclusion first drawn from it —
that the context axis is inert — was wrong.

### The value of long context is real, and concentrated in ~1% of tokens

`context_value.py` scores each token twice on sequences identical in length,
absolute positions and local context, differing only in whether the *distant*
context carries real information: bytes before offset `--corrupt-before` are
replaced with prose from another book. On `pg19-gate-16k-qknorm`, corrupting
everything before byte 8192 and scoring positions 12288+ (327,680 positions):

| statistic | value |
| --- | --- |
| mean gain | +0.0226 bpc |
| **median gain** | **+0.0002 bpc** |
| tokens helped / hurt | 54.2% / 45.8% |
| gaining > 1 bit | **0.96%** of tokens, carrying **79%** of all bits gained |
| top 1% of tokens | **+1.83 bpc each**, 81% of the total |

The mean agrees with the position-curve measurement (+0.024), so the two
independent methods corroborate each other. But the median is zero: the typical
byte gains nothing from 8K of extra context, and roughly 1% of tokens gain
~1.8 bits each. Averaging over the other 99% dilutes a large effect by ~80x.

**This changes what the study should measure, and revives it.** Mean bpc is a
poor instrument for a long-context question. On the subpopulation that actually
depends on distant retrieval, the effect is ~1.8 bpc — comfortably *larger*
than every confound found here (LR horizon 0.144, recency hole ~0.5, QK-norm
~0.7), rather than 5-45x smaller. The CSA question also sharpens: on the tokens
that need a distant block, does the lightning indexer's top-k select it? That is
a direct test of CSA's actual novel component.

#### Two measurement designs that failed first

Recorded because both produced confident nonsense — "gains" of 4+ bpc, larger
than the model's entire bpc:

- **Masking attention to a short window at eval.** A model trained with full
  attention is far off-distribution when attention is restricted.
- **Re-feeding only the recent bytes as their own sequence.** Worse: the
  sequence then has no position-0 token, and transformers depend on an
  early-token attention sink, so the model collapses to near-random (5.70 bpc
  against 6.98 for a uniform guess over this 126-value vocab).

Substituting real text from another document keeps length, positions and the
sink intact, so only information content changes.

**The first attempt diverged and answered nothing.** Gradient norms grew from
0.3 to 4.3e5, best val landed at step 2100 of 10000, and everything after that
degraded by ~50%; `train_bpc` tracked `val_bpc` throughout, so it was
optimization blowup rather than overfitting. The position curve it produced
came from a checkpoint 21% into the anneal and is not evidence about pg19.
Two fixes went in before any re-run:

- QK-RMSNorm in the dense baseline (above) — the root cause.
- The pg19 byte vocab is now built from **train counts only**, with values
  under `PG19_MIN_TRAIN_COUNT` folded into one `<rare>` id. Previously the
  vocab came from the union of splits, so 4 byte values appeared in test having
  never occurred in train, and 18 occurred fewer than 10 times — untrained
  embedding rows under weight tying, each occurrence producing a large gradient
  on a near-init row. Vocab 170 → 126; the folded tail is 0.0013% of the
  corpus. enwiki8 is deliberately left alone (min train count 2, nothing
  test-only) so its existing runs stay reproducible.

### What CSA is actually claiming

Worth stating plainly, because it sets the target. DeepSeek-V4 introduces CSA
"to enhance long-context **efficiency**" — the claim is cost, not quality. Their
own recipe (§4.2.2) makes this concrete: they **warm up with dense attention for
the first 1T tokens** and **introduce sparse attention only at 64K sequence
length**, having trained at 4K and 16K densely first.

So at matched parameters, matched tokens and matched context, dense attention is
close to an upper bound — CSA compresses `m` tokens into one entry and then
discards all but the top-k of those, and both steps are lossy. Expecting CSA to
*beat* dense in bpc under those constraints is expecting a lossy approximation
to beat what it approximates. The honest target is **parity**, and the honest
axis is quality per unit of compute or KV cache, neither of which this repo
currently measures.

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
| `--dataset` | `tinyshakespeare` | `tinyshakespeare`, `enwiki8`, or `pg19` |
| `--block-size` | 1024 | training sequence length |
| `--batch-size` | 32 | micro-batch |
| `--grad-accum-steps` | 1 | micro-batches per optimizer step |
| `--amp` | `none` | `none` or `bf16` (halves activation memory) |
| `--max-iters` | 3000 | stopping backstop |
| `--lr-horizon` | 0 | steps the LR anneal spans; 0 falls back to `--max-iters`. **Set this whenever early stopping is on** — see Results |
| `--lr-schedule` | `cosine` | `cosine` or `wsd` (flat, then linear decay over the last `--wsd-decay-frac`) |
| `--csa-chunk` | 0 | query-axis chunk for CSA attention; 0 = one shot. Memory only, results identical |
| `--csa-n-win` | 0 | sliding-window branch width (paper §2.3.3); 0 = off. DeepSeek-V4 uses 128 |
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
