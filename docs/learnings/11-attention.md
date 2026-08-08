# Learning 11 — Attention, worked all the way through

> **Date:** 2026-08-08 · **Context:** M2, after `attention_one_head`, before `multi_head_attention` · **Status:** living
>
> 📖 *Inference Engineering* §2.2.3 (attention, p.52)
> 🔧 **References:** [`src/forward.rs`](../../src/forward.rs) · `ds4` `flash_attn` / `softmax` · [HF `modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
> 🧭 Raschka: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — MHA, GQA, QK-norm, and RoPE in context

[Learning 05](05-reading-shapes.md) gives the head shapes, [Learning 08](08-row-major-strides.md)
gives their row-major storage, and [Learning 10](10-transformer-block-anatomy.md)
places attention inside a Qwen3 block. This note fills the missing middle: **what
attention computes**, with every intermediate number visible.

---

## The mental model: a content-addressed weighted read

Attention is not a vague instruction to “focus.” It is a **content-addressed
weighted read** from a table of values:

- **Q (query):** what does this token seek?
- **K (key):** how can each token be matched?
- **V (value):** what does that token carry if it is selected?

Q, K, and V are **different learned linear projections of the residual stream**.
They are not the raw token vectors, and K is not the payload: Q and K decide the
weights; V supplies the rows that those weights combine. This separation lets the
model learn one representation for matching and another for carried information.

## One head, with every axis explicit

For sequence length `seq` and head width `d`:

```text
residual projection gives Q, K, V       [seq, d]
scores S = Q Kᵀ / √d                    [seq, seq]
causal mask: S[t,j] = −∞ when j > t     [seq, seq]
A = row_softmax(S)                       [seq, seq]
O = A V                                  [seq, d]
```

Axis 0 is always the **token position**. In `Q/K/V/O`, axis 1 is a **head
feature**. In `S/A`, rows are query positions `t`, columns are key/value positions
`j`. Therefore output row `O[t,:]` mixes **value rows** `V[j,:]` using row
`A[t,:]`. It does not shuffle token order or merge output rows with one another:
there is still exactly one output row for each query position, in the original
sequence order.

The causal mask enforces autoregression: row `t` may use columns `0..t`, never a
future column. Implementations may materialize `−∞`, or—as `attention_one_head`
does—compute only the visible prefix. Those are the same math.

---

## Complete two-token trace (the exact Rust test)

The known-answer test in `src/forward.rs` uses:

```text
Q = K = [[1, 0],        V = [[10,  0],       seq = 2, d = 2
         [0, 1]]             [ 0, 20]]
```

### 1. Dot products, then scale

`QKᵀ = [[1,0],[0,1]]`. Since `1/√d = 1/√2 = 0.70710678…`:

```text
S = QKᵀ / √2 = [[0.7071, 0     ],   shape [2,2]
                [0,      0.7071]]
```

Why divide by `√d`? A dot product sums `d` terms. As `d` grows, its typical
magnitude grows too, pushing softmax toward saturated, near-one-hot probabilities
and poor gradients. `1/√d` keeps score scale roughly comparable across head widths.

### 2. Causal mask

The upper-right score is the score from query position `t=0` to future key
position `j=1`; replace it with `−∞`:

```text
S_masked = [[0.7071, −∞    ],
            [0,       0.7071]]
```

### 3. Stable row-wise softmax

For each row, subtract its maximum before exponentiating:

```text
softmax(x)i = exp(xi − max(x)) / Σj exp(xj − max(x))
```

Subtracting one constant from every score leaves probabilities unchanged, but
prevents `exp(large_score)` from overflowing. Masked `−∞` exponentiates to zero.

```text
A = [[1,          0         ],
     [0.33023846, 0.66976154]]   shape [2,2]
```

For row 1, subtracting `0.70710678` gives `[-0.70710678, 0]`; normalized
exponentials are exactly the two weights above.

### 4. Weighted value read

```text
O = A V
  = [[1·[10,0] + 0·[0,20]],
     [0.33023846·[10,0] + 0.66976154·[0,20]]]
  = [[10,        0        ],
     [ 3.3023846, 13.395231]]                    shape [2,2]
```

At `t=0`, the causal mask means the query **cannot see `v1`**, regardless of its
score; output row 0 is exactly `v0`. At `t=1`, both positions are visible, so output
row 1 mixes both values—about 33% of `v0` and 67% of `v1`.

---

## From one head to Qwen3's heads

Learned projections produce **packed heads**. We split their feature axis into
contiguous `d`-wide slices, run each query head independently, concatenate all
query-head outputs, then `o_proj` maps back to residual width `H`.

For Qwen3-0.6B (`H=1024`, 16 query heads, 8 KV heads, `d=128`):

```text
h                         [seq, 1024]
q_proj(h)                 [seq, 2048] = [seq, 16·128]
k_proj(h), v_proj(h)      [seq, 1024] = [seq,  8·128]
16 attention outputs,
  concatenated            [seq, 2048]
o_proj                    [seq, 1024]  → back to H
```

Each query head has its own query slice and independently computes its own score
matrix, row-softmax, and weighted value read. Concatenation preserves those
distinct results until `o_proj` learns how to combine them.

### GQA is K/V sharing, not query averaging

Grouped-query attention has `group = query_heads / kv_heads = 16/8 = 2`.
Query head `h` uses KV head `floor(h/2)`:

```text
q0,q1 → kv0   q2,q3 → kv1   …   q14,q15 → kv7
```

The two query heads in a group reuse the same K and V rows, reducing KV storage
and later KV-cache bandwidth. They **are not averaged**, and they do not share an
output: their different Q vectors produce different score matrices and therefore
different weighted reads, even against the same K/V.

---

## Exact Qwen3 attention order

Order is part of the architecture; changing it changes the numbers:

```text
residual h
  → q_proj / k_proj / v_proj
  → split packed features into heads
  → RMSNorm each q head and each k head over d       (not V)
  → RoPE each q head and each k head                  (not V)
  → per-query-head causal attention, sharing K/V by GQA
  → concatenate query-head outputs
  → o_proj → width H
```

**QK-norm and `1/√d` both happen, for different reasons.** QK-norm is a learned
per-feature RMS normalization of each Q/K vector that controls its representation
and magnitude. Scaling is a fixed factor on every resulting dot product that
accounts for the sum across `d` features. One does not replace the other.

Likewise, **RoPE and the causal mask solve different problems**. RoPE rotates Q/K
so their match score carries relative-position information—*where* tokens are
relative to each other. The mask enforces information flow—*which* positions are
legal to read. A position-aware score may still point into the future; the mask
must still forbid it.

---

## Bridge to the implementation

`src/forward.rs::attention_one_head` is now implemented. It asserts `q/k/v
[seq,d]`, computes only each causal prefix, uses max-subtracted stable softmax, and
returns `[seq,d]`. Its two-token test is the complete trace above; shape tests fail
loudly. Next is `multi_head_attention`: projections, head slices, QK-norm, RoPE,
GQA mapping, concatenation, and `o_proj`. The assembled block 0 will then be checked
against the committed official HF golden checkpoint before all 28 layers proceed.

### Common misconceptions checklist

- [ ] “Attention means focus.” → It is a precise content-addressed weighted read.
- [ ] “Q/K/V are token embeddings.” → They are different learned projections of the residual stream.
- [ ] “K is what gets copied.” → K is matched; V is carried.
- [ ] “Softmax runs over the whole matrix.” → It runs independently across each query row.
- [ ] “The output reorders tokens.” → Every query keeps its row; only value content is mixed into it.
- [ ] “GQA averages query heads.” → Query heads stay independent; only K/V heads are shared.
- [ ] “QK-norm replaces `1/√d`.” → Learned vector normalization and fixed score scaling both happen.
- [ ] “RoPE makes the causal mask unnecessary.” → Position encoding and visibility constraints are separate.

## Cross-links

- ⬅ [Learning 05 · reading shapes](05-reading-shapes.md) — head width and GQA projection shapes.
- ⬅ [Learning 08 · row-major & strides](08-row-major-strides.md) — how packed rows and head slices live in memory.
- ⬅ [Learning 10 · transformer block anatomy](10-transformer-block-anatomy.md) — attention's place on the residual bus.
- ▶ [Interactive single-head toy](../diagrams.html#attention) — vary the query and causal mask.
- 🔧 [`src/forward.rs`](../../src/forward.rs) — the literal prefix-only one-head loop and its known-answer test.
- 🔧 `ds4` — `flash_attn` and `softmax` are the fused/accelerated descendants of this same math.
- 🔧 [HF `modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) — architecture spec and golden oracle.
- 📖 *Inference Engineering* (Kiely), §2.2.3 · 🧭 [Raschka's architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison).
