# mini-deepseek-v4

A small-scale, locally-trainable PyTorch implementation of just the **Compressed
Sparse Attention (CSA)** mechanism from the DeepSeek-V4 paper
(*Towards Highly Efficient Million-Token Context Intelligence*, 2026), §2.3.1,
eqs. (9)–(19).

The goal is **architectural understanding and validation**, not a frontier model.
A tiny vanilla-attention baseline and a CSA variant are trained on the same
small dataset and their loss curves compared. See [`notes.md`](notes.md) for the
full dimension table and the six approved design decisions (D1–D6).

## Project status

Built incrementally; each stage is committed in a runnable state.

- [x] **Stage 1** — vanilla decoder-only transformer baseline.
- [x] **Stage 2** — CSA compression (eqs. 9–12) with dense attention over all C^Comp blocks.
- [ ] Stage 3 — Lightning indexer (eqs. 13–17).
- [ ] Stage 4 — Shared-KV MQA (eqs. 18–19).

## Results

Both runs use the same architecture skeleton: `d_model=384`, 6 layers, 6 heads
(`head_dim=64`), `block_size=1024`, `batch_size=32`, AdamW lr 3e-4 → 3e-5
cosine schedule, 100-step warmup, 3000 iters, char-level TinyShakespeare,
tied LM head. The only thing that varies is the attention block.

CSA Stage 2 uses `m=4`, `c=64` (= head_dim), so the KV cache is compressed
4× along the sequence dimension. Queries attend densely over all causally-
valid compressed blocks (no indexer or top-k yet — that arrives Stage 3).

| metric                          | baseline-v1   | csa-stage2-v1   |
| ------------------------------- | ------------- | --------------- |
| init loss                       | 4.18          | 4.18 (= ln 65)  |
| **min val loss (step)**         | **1.585 (1000)** | **1.977 (1500)** |
| final train loss (step 3000)    | 0.151         | 1.378           |
| final val loss (step 3000)      | 3.116         | **2.101**       |
| train/val gap @ step 3000       | 2.965 nats    | **0.723 nats**  |
| params (excl embeddings)        | ~10.7M        | ~10.5M          |
| throughput                      | 217K tok/s    | 192K tok/s\*    |
| wall time                       | 528 s         | 589 s           |

![baseline vs csa-stage2](results/baseline_vs_csa-stage2.png)

\* Throughput is *worse* than baseline despite CSA having ~4× fewer attention
FLOPs because Stage 2 uses an explicit `masked_fill + softmax + nan_to_num`
path rather than `F.scaled_dot_product_attention` (which the baseline gets to
use). Vectorizing the safe-softmax against SDPA is a Stage 4 cleanup item;
in v1 we prioritize clarity over perf (per project plan).

### Reading the curves

- Baseline overfits hard: train loss → 0.15 (memorization), val loss bottoms
  at step 1000 and *climbs* to 3.12 by step 3000.
- CSA's compression is a strong regularizer: train loss only drops to 1.38,
  val loss bottoms higher (1.98) but *stays nearly flat* (2.10 at step 3000).
- The two val curves cross around step 1750 — past that point CSA's val loss
  is *better* than baseline's, purely because baseline is overfitting.
- Best-case CSA val loss is ~0.4 nats worse than best-case baseline. That's
  the cost of 4× compression on a fine-grained char-level task. A generation
  sample shows CSA produces Shakespeare-like structure (newlines, character-
  name colons, dialogue rhythm) but words are rougher than baseline's,
  consistent with the loss gap.

### Eval comparability note

CSA eval excludes the first `m=4` positions per sequence from the loss (D2 in
`notes.md` — those positions have no causally-valid compressed block).
Baseline includes all 1024 positions in eval. The 4-of-1024 = 0.4% gap is
documented but not corrected: it's well below the 0.4-nat differences we
care about, and re-evaluating baseline with the same mask wouldn't change
the qualitative conclusions.

## Setup

Requires Python 3.12 and a CUDA GPU (tested on RTX 5070 Ti, sm_120, CUDA 12.8).

```bash
uv venv --python 3.12
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install -r requirements.txt
```

(`torch` is installed from the CUDA-12.8 wheel index because Blackwell-class GPUs
need it; CPU-only or older-CUDA setups can use the default index.)

## Run

```bash
# vanilla baseline
.venv/bin/python train.py --run-name baseline-v1 --max-iters 3000

# CSA (Stage 2+)
.venv/bin/python train.py --run-name csa-v1 --attention csa --max-iters 3000

# plot
.venv/bin/python plot.py baseline-v1                # single run
.venv/bin/python plot.py baseline-v1 csa-v1         # comparison
```

Logs land in `runs/<run-name>/log.jsonl` (jsonl, one entry per train/eval event),
checkpoints in `runs/<run-name>/final.pt` (gitignored — too large), and plots in
`results/`.

## Configuration knobs

`train.py --help` lists everything. The key knobs:

| Flag             | Default | What                                              |
| ---------------- | ------- | ------------------------------------------------- |
| `--attention`    | vanilla | `vanilla` or `csa`                                |
| `--d-model`      | 384     | hidden size                                       |
| `--n-layers`     | 6       | transformer blocks                                |
| `--n-heads`      | 6       | attention heads                                   |
| `--block-size`   | 1024    | training sequence length                          |
| `--batch-size`   | 32      |                                                   |
| `--max-iters`    | 3000    |                                                   |
| `--lr`           | 3e-4    | peak learning rate (cosine schedule, warmup 100)  |

## Design

Architecture choices for v1, all justified in [`notes.md`](notes.md):

- **No RoPE** — learned absolute position embeddings, identical for baseline
  and CSA, so the comparison isolates the attention mechanism.
- **RMSNorm pre-norm**, **SwiGLU MLP** — standard modern decoder ingredients.
- **Char-level tokenizer** on TinyShakespeare (vocab=65). Simpler than BPE for
  a learning project.
- **Tied LM head** weights with token embeddings.
- **AdamW**, betas (0.9, 0.95), weight decay 0.1 on 2D params only, grad clip 1.0,
  cosine LR schedule with linear warmup.

### Known limitations (v1)

- The first `m` positions of every sequence are **masked from the loss** in the
  CSA variant (no causally-valid compressed block exists). The model is
  technically untrained on those positions; eval excludes them. See D2 in
  `notes.md`.
- **Train/eval mismatch**: indexer is dense at train, top-k at eval (D3). Eval
  reports *both* numbers. Don't claim "CSA matches baseline" using only the
  dense number.
- **No length extrapolation** — fixed `block_size`, no RoPE.

### Out of scope (do not add)

Per the project plan: MoE, FP4/FP8, Muon optimizer, mHC residuals, HCA,
sliding-window attention, attention sink, multi-GPU, gradient checkpointing,
FlashAttention. Each is its own project.

## Repo layout

```
model.py            # transformer + (later) CSA attention block
train.py            # training loop, --attention {vanilla,csa}
data.py             # TinyShakespeare char-level loader
plot.py             # loss-curve plotting
tests/              # unit tests for CSA math (Stage 2+)
notes.md            # dimension table + design decisions D1-D6
runs/<name>/        # per-run logs, configs, checkpoints
results/            # plots
```
