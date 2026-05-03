# Mini DeepSeek-V4 CSA — Design Notes

Reference: DeepSeek-V4 paper, **Section 2.3.1** (pp. 9–11), eqs. (9)–(19). The paper's open-source reference is at `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`.

---

## Symbol legend

| Symbol      | Meaning                                              | Typical value (this project) |
| ----------- | ---------------------------------------------------- | ---------------------------- |
| `B`         | batch size                                           | 8–32                         |
| `n`         | sequence length (must be a multiple of `m`)          | 1024–2048                    |
| `d`         | model hidden dim                                     | 256–512                      |
| `m`         | compression factor (KV tokens per compressed block)  | 4 or 8                       |
| `n_blk`     | number of compressed blocks = `n / m`                | 128–512                      |
| `c`         | core attention head dim (also dim of `C^Comp`)       | 64–128                       |
| `c^I`       | indexer head dim                                     | 32–64                        |
| `n_h`       | core query heads                                     | 4–8                          |
| `n_h^I`     | indexer query heads                                  | 2–4                          |
| `d_c`       | shared query latent dim (used by indexer + core)     | 128–256                      |
| `k`         | top-k compressed blocks selected per query (eval)    | 16–32                        |

Convention: shapes use einops-style names (`b n d`). All tensors below are per-batch unless noted.

---

## Stage 1 — Compressed KV entries (eqs. 9–12)

### Parameters

| Name                     | Shape       | Role                                                                |
| ------------------------ | ----------- | ------------------------------------------------------------------- |
| `W^{aKV}`, `W^{bKV}`     | `(d, c)`    | Project hidden state to two KV streams `C^a`, `C^b`                 |
| `W^{aZ}`,  `W^{bZ}`      | `(d, c)`    | Project hidden state to two compression-weight streams `Z^a`, `Z^b` |
| `B^a`,     `B^b`         | `(m, c)`    | Learnable positional biases inside a compressed block               |

### Activations

| Name           | Shape          | Source                                  | Notes                                          |
| -------------- | -------------- | --------------------------------------- | ---------------------------------------------- |
| `H`            | `(b, n, d)`    | block input                              |                                                |
| `C^a`, `C^b`   | `(b, n, c)`    | eq. 9                                    | per-token KV streams                           |
| `Z^a`, `Z^b`   | `(b, n, c)`    | eq. 10                                   | per-token compression-weight streams           |
| `C^Comp`       | `(b, n_blk, c)`| eqs. 11–12                               | one row per compressed block                   |

### Per-block computation (block `i ∈ [0, n_blk)`)

Define windows:
- "current" window covers token indices `[mi, m(i+1))` — used from `C^a`/`Z^a`
- "previous" window covers token indices `[m(i-1), mi)` — used from `C^b`/`Z^b`
- Boundary `i = 0`: previous window is **out of range** → pad `Z^b` slice with `-inf`, pad `C^b` slice with zeros.

| Intermediate                           | Shape        |
| -------------------------------------- | ------------ |
| `Z^a` window + `B^a`                   | `(b, m, c)`  |
| `Z^b` window + `B^b`                   | `(b, m, c)`  |
| Concatenated logits along block axis   | `(b, 2m, c)` |
| `S = Softmax_row(...)`                 | `(b, 2m, c)` |
| `S^a` (top half), `S^b` (bottom half)  | `(b, m, c)`  |

**Softmax direction (eq. 11) — important.** "Softmax_row" in the paper means *normalize across the row dimension*: for each of the `c` feature columns independently, softmax over the `2m` row positions. Each `(b, c)` column sums to 1 over the 2m axis. (Verbatim: *"performs normalization across the total of 2m elements from both Z^a and Z^b"*.)

This is **per-feature pooling**, not standard attention softmax — each feature dim independently picks how to blend the 2m source positions.

| Output                                          | Shape         | Equation                                           |
| ----------------------------------------------- | ------------- | -------------------------------------------------- |
| `C_i^Comp = Σ_{j ∈ cur} S^a_j ⊙ C^a_j + Σ_{j ∈ prev} S^b_j ⊙ C^b_j` | `(b, c)` | eq. 12 |

### Invariant to test

> "indexes of `C^b` used for `C_i^Comp` and indexes of `C^a` used for `C_{i-1}^Comp` are overlapped" — both cover token range `[m(i-1), mi)`, but via different projections (`a` = current view, `b` = previous view). Adjacent compressed blocks share token *ranges*, not values.

---

## Stage 2 — Lightning indexer (eqs. 13–17)

### Parameters

| Name        | Shape                  | Role                                                  |
| ----------- | ---------------------- | ----------------------------------------------------- |
| `W^{DQ}`    | `(d, d_c)`             | down-project hidden state to **shared** query latent  |
| `W^{IUQ}`   | `(d_c, c^I · n_h^I)`   | up-project latent into `n_h^I` indexer query heads    |
| `W^w`       | `(d, n_h^I)`           | per-token, per-head indexer scoring weights           |
| `W^{aKV,I}`, `W^{bKV,I}`, `W^{aZ,I}`, `W^{bZ,I}` | `(d, c^I)` | indexer-side compressor (separate from main) |
| `B^{a,I}`,  `B^{b,I}`  | `(m, c^I)`             | indexer-side positional biases                        |

### Activations

| Name        | Shape                  | Source                                            |
| ----------- | ---------------------- | ------------------------------------------------- |
| `c^Q`       | `(b, n, d_c)`          | eq. 13 — `H · W^{DQ}`. **Shared with Stage 3.**   |
| `q^I`       | `(b, n, n_h^I, c^I)`   | eq. 14 — `c^Q · W^{IUQ}`, reshape to heads        |
| `w^I`       | `(b, n, n_h^I)`        | eq. 15 — `H · W^w`                                |
| `K^{IComp}` | `(b, n_blk, c^I)`      | same compressor as `C^Comp`, separate weights     |
| `I` scores  | `(b, n, n_blk)`        | eq. 16 — see below                                |

### Score computation (eq. 16)

For each query `t` and compressed block `s`:
```
I[t,s] = Σ_{h=1..n_h^I}  w^I[t,h] · ReLU(  q^I[t,h] · K^{IComp}[s]  )
                                          ^^^ inner product in c^I
```

### Causal mask

Query `t` may only score blocks `s < floor(t / m)` (paper's eq. 16 condition). Mask `I[t, s ≥ floor(t/m)] = -inf`.

**Consequence:** queries with `t < m` (the very first compressed block) have **zero** valid blocks. The paper handles this with sliding-window + attention-sink branches, both out of scope here. See "Open decisions" below.

### Top-k selection (eq. 17)

```
C^SprsComp[t] = { C^Comp[s] : I[t, s] ∈ Top-k(I[t, :]) }    shape (b, n, k, c)
```

`Top-k` is non-differentiable. **Training:** dense over all `n_blk` blocks (skip top-k). **Eval:** apply top-k. (Choice documented below.)

---

## Stage 3 — Shared-KV MQA (eqs. 18–19)

### Parameters

| Name        | Shape               | Role                                                   |
| ----------- | ------------------- | ------------------------------------------------------ |
| `W^{UQ}`    | `(d_c, c · n_h)`    | up-project shared latent `c^Q` into `n_h` core heads   |
| `W^O`       | `(c · n_h, d)`      | final output projection (skip grouped variant for v1)  |

### Activations

| Name      | Shape                  | Source                                              |
| --------- | ---------------------- | --------------------------------------------------- |
| `q`       | `(b, n, n_h, c)`       | eq. 18 — `c^Q · W^{UQ}`, reshape                    |
| K = V     | `(b, n, k, c)`         | gathered from `C^Comp` via top-k indices (or all `n_blk` during dense training) |
| `o`       | `(b, n, n_h, c)`       | eq. 19 — per-head MQA: each head uses same K = V    |
| `out`     | `(b, n, d)`            | `o.flatten(-2) · W^O`                                |

**Key MQA property:** all `n_h` query heads share **one** K = V tensor (no per-head K/V projections). K and V are literally the same tensor (the gathered compressed entries). This is the major FLOPs/cache win.

---

## Param-count sanity check (one CSA layer, n_h=4, n_h^I=2, d=384, c=64, c^I=32, d_c=192)

| Block                   | Params (≈) |
| ----------------------- | ---------- |
| Main compressor (4 × `d×c` + 2 × `m×c`) | 98 K |
| Indexer compressor (4 × `d×c^I` + 2 × `m×c^I`) | 49 K |
| `W^{DQ}`                                | 74 K  |
| `W^{IUQ}` (`d_c × c^I·n_h^I`)           | 12 K  |
| `W^{UQ}` (`d_c × c·n_h`)                | 49 K  |
| `W^w` (`d × n_h^I`)                     | 0.8 K |
| `W^O` (`n_h·c × d`)                     | 98 K  |
| **Total per CSA layer**                 | **~380 K** |

For 6 layers + embeddings (vocab≈50k, d=384) ≈ 19M model. Within budget.

---

## Invariants the unit tests should assert (Stage 1)

Hand-computed test, `m=2`, `c=2`, `n=4`, batch=1:

1. `C^a.shape == C^b.shape == Z^a.shape == Z^b.shape == (1, 4, 2)`
2. For block `i=1`: stacked `(2m=4, c=2)` matrix → softmax columns sum to 1 (each of 2 columns sums to 1.0 across the 4 rows).
3. For block `i=0`: `Z^b` slice padded with `-inf` → after softmax those rows have weight 0 → `C_0^Comp` reduces to `Σ S^a_j ⊙ C^a_j` only.
4. `C^Comp.shape == (1, 2, 2)` (n_blk = 4/2 = 2).
5. With identity-like weights (`W^{aZ}=0, W^{bZ}=0` so `Z^a=Z^b=0`, biases zeroed too), uniform softmax gives `C_i^Comp = mean(C^a window) + mean(C^b window)` (for `i ≥ 1`) — easy hand check.
6. Overlap invariant: token range `[m(i-1), mi)` enters both `C_i^Comp` (via `C^b`) and `C_{i-1}^Comp` (via `C^a`).

Stage 2 tests:
7. `I[t, s] == 0` whenever every `q^I[t,h]·K^{IComp}[s]` is negative (ReLU kills it).
8. Causal mask: `I[t, s] == -inf` for all `s ≥ floor(t/m)`.

---

## Decisions (approved 2026-05-02)

These are the design contract for v1. Each entry was an open question; all six are now resolved. Paper-faithful trajectory items moved to the "Future work" section below.

### D1 — Indexer compressor weight sharing → **SEPARATE WEIGHTS**

The indexer uses its own `W^{aKV,I}, W^{bKV,I}, W^{aZ,I}, W^{bZ,I} ∈ R^{d × c^I}` and `B^{a,I}, B^{b,I} ∈ R^{m × c^I}`. Output dim differs (`c` vs `c^I`) so projections cannot be identical, and the indexer is meant to be cheap-and-different — sharing would tie the two together and defeat the design.

### D2 — First-`m`-tokens problem → **MASK FROM LOSS**

For positions `t ∈ [0, m)`, the indexer's causal constraint `s < floor(t/m)` admits zero blocks. We zero out the cross-entropy loss at these positions during training (per-sequence, the first `m` positions contribute no gradient).

> **Known limitation:** the model is therefore *technically untrained* on positions `0..m-1` of every sequence. At eval, behavior on these positions is undefined — they must be excluded from any reported metric. README must call this out. Sliding-window or sink tokens would address this and are explicit future work.

### D3 — Indexer training signal → **DENSE-AT-TRAIN, TOP-K-AT-EVAL**

During training, core attention runs densely over all causally-valid compressed blocks. Indexer scores `I[t,s]` are computed and trained but **not** used as a hard top-k mask. At eval, top-k is applied.

> **Caveat — train/eval mismatch (must report).** The model never sees the top-k bottleneck during training, so eval-with-top-k may be worse than the dense-train loss curves suggest. The eval script reports **both**:
> - dense eval loss/perplexity (matches train regime)
> - top-k eval loss/perplexity at the chosen `k` (matches deployment regime)
>
> Compare both to the vanilla baseline. Do not claim "CSA matches baseline" using only the dense number.

### D4 — Normalization and positional encoding → **RMSNorm YES, NO RoPE IN V1**

- **RMSNorm:** include. Apply per-head to `q[t, i]` and once to each `C^Comp[s]` before core attention (paper §2.3.3). Standard, cheap, helps stability.
- **Positional encoding:** **no RoPE anywhere in v1**, in either the baseline or the CSA path. Use **learned absolute position embeddings** added to input embeddings, identical for both architectures.
  - *Why no RoPE:* applying RoPE to compressed entries `K^IComp[s]` / `C^Comp[s]` is non-trivial — they don't correspond to a single token position, and the paper's fix (partial RoPE on last 64 dims + negative-position trick on outputs) is exactly the complexity scoped out under D4's "skip partial RoPE" intent.
  - *Why "no RoPE in baseline either":* fairness. Both architectures get the same positional information source. A RoPE-baseline vs. abs-pos-CSA comparison would conflate "positional encoding choice" with "attention mechanism".
  - *Cost:* at fixed `n = 2048` we lose only length-extrapolation ability, which is irrelevant for v1. Documented as a deliberate deviation from both modern practice and the paper.

### D5 — Output projection → **PLAIN W^O, NO GROUPING**

Standard `W^O ∈ R^{c·n_h × d}` linear. Grouped variant is a FLOPs optimization for production, not a correctness concern at our scale.

### D6 — Attention sink → **SKIPPED**

Solves a different problem (long-context stability) than what we're testing.

---

## Future work (paper-faithful trajectory, explicitly out of v1)

Tracked here so the gap between v1 and the paper is visible, not as a backlog to grind through.

- **Auxiliary KL loss for the indexer** (paper's approach): use dense attention's distribution as a teacher for the top-k student. Removes the train/eval mismatch from D3.
- **Partial RoPE with negative-position trick on outputs** (paper §2.3.3). Required for length extrapolation.
- **Sliding-window attention branch** — would naturally cover the first-`m`-tokens hole from D2.
- **Learnable attention-sink logits** (paper eq. 27).
- **Grouped output projection** — when `c·n_h` grows large.
- **Heavily Compressed Attention (HCA)** layers and hybrid CSA/HCA configuration (paper §2.3.2).
- **`m`HC residuals, MoE, FP4, Muon** — outside the attention-mechanism focus entirely.

---

## Stage plan recap (for the implementation phase)

1. **Stage 1 — vanilla baseline transformer.** Standard MHA, learned absolute position embeddings (per D4), RMSNorm pre-norm, AdamW. Train on TinyShakespeare. Save loss curve.
2. **Stage 2 — compression only.** Implement eqs. 9–12. Replace standard K/V with `C^Comp`. Queries attend densely over all `n_blk` compressed entries (no indexer yet). Unit tests for compression math. Train + compare to baseline.
3. **Stage 3 — lightning indexer.** Add eqs. 13–17. Use D3 proposal (dense at train, top-k at eval). Unit tests for indexer scoring + causal mask.
4. **Stage 4 — Shared-KV MQA.** Add eqs. 18–19 properly. The MQA wiring is mostly already there from Stage 2 since K=V were already shared; this stage is about confirming the up-projection from the shared latent works and the per-head core attention is correct.

Each stage gets its own commit and runnable state. (Repo is not yet a git repo; will `git init` at start of Stage 1 unless you want it earlier.)
