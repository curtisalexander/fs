# Learning 05 — Reading shapes: how the dimensions line up

> **Date:** 2026-06-24 · **Context:** M1, loading the weights · **Status:** living
>
> 📖 *Inference Engineering* §2.1 (the transformer block, p.42), §2.2.2–2.2.3
> 🔧 `ds4`: tensor shapes in `ds4.c` + `metal/{dense,flash_attn,glu,norm}.metal`
> 🧭 Raschka: ["Understanding LLMs" architecture comparison](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — *inspect the config, line up the dims*

The single hardest part of reading (or writing) neural-net code isn't the math —
it's keeping the **shapes** straight. Which axis is which? Why is the weight shape
"backwards" from the data? Where did `2048` come from when `hidden_size` is `1024`?
This note is the map we come back to every time a tensor shows up. **Clarity
first** — once we can *see* the dimensions line up, the code (and later the speed
work) is much easier.

We use Qwen3-0.6B's real numbers throughout, straight from its `config.json`.

---

## The one rule

A matrix multiply `A · B` is legal **iff the inner dimensions match**:

```
A: (m × k)   B: (k × n)   ⟹   A·B: (m × n)
        └────────┘
        these two k's (the "inner" dims) must be equal — k is then
        "contracted" (summed away); the outer dims m and n survive.
```

That's it. Every shape question below is an application of this one rule. When a
forward pass breaks, 9 times out of 10 it's two `k`s that didn't match — so we
make the `k`s **visible** and **assert** them.

## The #1 gotcha: weights are stored `[out, in]`

A linear layer computes `y = x · Wᵀ`. In PyTorch (and therefore in the
safetensors file) the weight is stored **transposed** relative to the data flow,
as `[out_features, in_features]`:

```
x            W (stored [out, in])         y
(1 × in)     [out × in]                   (1 × out)

         in ─────────────▶ out
   x ──▶ [ · · · · · · · · ]  W  ──▶  y
         ▲ contract this axis
         must equal x's width (in)
```

So when you read a row of the tensor table:

```
self_attn.q_proj.weight   [2048, 1024]
                           └out┘  └in┘
```

read it as **`in=1024 ──▶ out=2048`**: it takes a 1024-wide vector and produces a
2048-wide one. The stored shape looks reversed from the arrow because of the `Wᵀ`.
Internalize this once and the whole table becomes readable.

## The residual stream: a fixed-width bus

A transformer block doesn't change the width of the thing flowing through it.
Tokens enter as `H`-wide vectors, every sub-layer reads from and writes back to
that same `H`-wide **residual stream**, and the block outputs `H`-wide vectors.
For Qwen3-0.6B, **`H = hidden_size = 1024`**. Hold that constant in your head:

```
        ┌────────── one block (× 28) ──────────┐
…──H──▶ │ norm → attn → (+) → norm → FFN → (+) │ ──H──▶ …
        └──────────────────────────────────────┘
         every "(+)" folds the sub-layer output back into the H-wide bus
```

Attention and the FFN *fan out* to wider working widths internally, then *project
back* to `H` before rejoining the bus. The fan-out widths are where the other
dimensions come from.

## Qwen3-0.6B's named dimensions (the legend)

Every shape in the model is built from these seven numbers:

| symbol | name (`config.json`)      | value   | what it is |
|:------:|---------------------------|--------:|------------|
| `V`    | `vocab_size`              | 151936  | how many distinct tokens |
| `H`    | `hidden_size`             | 1024    | residual-stream width |
| `L`    | `num_hidden_layers`       | 28      | transformer blocks |
| `d`    | `head_dim`                | 128     | width of one attention head |
| —      | `num_attention_heads`     | 16      | query heads → `16·d = 2048` |
| —      | `num_key_value_heads`     | 8       | key/value heads → `8·d = 1024` |
| `I`    | `intermediate_size`       | 3072    | FFN inner width |

Two of these deserve their own diagram because they're the usual sources of
confusion.

### head_dim is *decoupled* from hidden_size

Naively you'd expect `num_heads × head_dim == hidden_size`. **It doesn't here:**
`16 × 128 = 2048 ≠ 1024`. Qwen3 lets the attention working width differ from the
residual width. So `q_proj` projects `H ──▶ 16·d`, i.e. `[2048, 1024]`, and after
attention `o_proj` projects back `16·d ──▶ H`, i.e. `[1024, 2048]`:

```
 H=1024 ──q_proj [2048,1024]──▶ 2048 = 16 heads × 128
                                 │  split into heads
                ┌────┬────┬─ … ─┬────┐   16 slices of width d=128
                │h0  │h1  │     │h15 │
                └────┴────┴─ … ─┴────┘
 attention happens per head, then concat ──o_proj [1024,2048]──▶ H=1024
```

A reader that "knows" `hidden == heads × head_dim` will mis-shape `q_proj`. The
table and the asserts exist precisely to catch that.

### GQA: query heads share key/value heads

Grouped-Query Attention uses **fewer key/value heads than query heads**. Here:
16 query heads, 8 kv heads → a **group size of `16 / 8 = 2`** (every 2 query heads
share one kv head). That's why `q_proj` is width `2048` but `k_proj`/`v_proj` are
width `1024 = 8·d`:

```
q heads:  Q0 Q1  Q2 Q3  Q4 Q5  …  (16, width 2048)
            \ /    \ /    \ /
kv heads:   KV0    KV1    KV2   …  ( 8, width 1024)   group = 2
```

(Saving K/V is the whole point — it shrinks the KV cache at M4. For M1 it just
explains the asymmetric shapes.)

## Walking one block, width by width

Now the full tensor set of a block reads as a chain of `in ──▶ out` arrows, every
arrow obeying the one rule. (`norm` weights are 1-D **scale vectors**, not
matrices — they multiply element-wise, so their width just states which thing they
scale.)

```
embedding (global)   embed_tokens [V, H]        token id ─▶ H        (a row lookup)

── block × 28 ───────────────────────────────────────────────────────────────
 input_layernorm        [H]            scale the H-wide bus
 q_proj  [16·d, H] = [2048, 1024]      H      ─▶ 16·d
 k_proj  [ 8·d, H] = [1024, 1024]      H      ─▶  8·d
 v_proj  [ 8·d, H] = [1024, 1024]      H      ─▶  8·d
 q_norm  [d] = [128]                   scale each query head (width d)
 k_norm  [d] = [128]                   scale each key   head (width d)
 o_proj  [H, 16·d] = [1024, 2048]      16·d   ─▶ H        (back onto the bus)
 post_attention_layernorm [H]          scale the H-wide bus
 gate_proj [I, H] = [3072, 1024]       H      ─▶ I    ┐ SwiGLU:
 up_proj   [I, H] = [3072, 1024]       H      ─▶ I    │ act(gate) * up,
 down_proj [H, I] = [1024, 3072]       I      ─▶ H    ┘ then back to the bus

── final ─────────────────────────────────────────────────────────────────────
 model.norm [H]                        scale the H-wide bus
 lm_head  (tied → embed_tokens)        H ─▶ V    reuse the embedding table
```

Notice the rhythm: **everything leaves the `H`-wide bus, does work at `16·d` /
`8·d` / `I`, and comes back to `H`.** If you can see that, you can read any dense
transformer's weights.

### Tied embeddings — one table, two jobs

`tie_word_embeddings: true` means there is **no separate `lm_head.weight`**. The
`[V, H]` embedding table is used twice: as a row lookup on the way in (token id ─▶
`H`), and transposed as the output projection on the way out (`H` ─▶ `V` logits).
That one table is `151936 × 1024 ≈ 155.6M` params — **~26% of the model's ~596M**.
A quarter of "the weights" is just the vocabulary.

## How this shows up in `fs inspect` (and in the code)

`fs inspect` is built to make all of the above legible at a glance:

- a **dimension legend** (the table above) printed first, so every shape has a
  named source;
- a grouped-by-layer tensor table whose last column is the **`in ──▶ out`** arrow,
  not just the raw shape;
- a **cross-check** that asserts the shapes line up with the config — every
  Linear's `in` equals the width feeding it (`q/k/v ← H`, `o ← 16·d`,
  `gate/up ← H`, `down ← I`), `embed = [V, H]`, lm_head tied — and reports total
  params. These asserts are the M1 verification, and they carry into M2 so a
  mis-wired matmul fails *loudly* instead of producing quiet garbage.

> **Learning-first, then fast.** Right now we assert shapes everywhere and convert
> nothing we don't have to. Making the math go fast (fusing, batching, skipping
> checks on the hot path) is a *separate* lesson later — and itself a good one.

---

## Cross-links

- ⬅ [`learning 01 · safetensors vs GGUF`](01-safetensors-vs-gguf.md) — where these
  shapes physically live on disk (the `[out, in]` blob).
- ➡ `docs/02-weights.md` — the M1 milestone writeup (uses this legend).
- 🔧 `ds4`: `metal/dense.metal` (the matmul), `metal/flash_attn.metal` (head
  layout), `metal/glu.metal` (SwiGLU), `metal/norm.metal` (RMSNorm scale vectors).
- 🧭 Raschka's architecture comparison — the GQA / head_dim choices across models.
