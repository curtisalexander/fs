# Learning 10 — Anatomy of a transformer block (and how we *know*)

> **Date:** 2026-07-14 · **Context:** M1, building `expected_tensors` · **Status:** living
>
> 📖 *Inference Engineering* §2.1 (neural nets, p.42), §2.2.1 (LLM architecture, p.49), §2.2.2–2.2.3 (blocks + attention)
> 🔧 **Reference:** [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) · **Design:** [Qwen3 Technical Report, arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
> 🧭 Raschka: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — QK-norm, GQA, SwiGLU in context

[Learning 05](05-reading-shapes.md) told us the *shapes* — how the dimensions line
up. This note answers the two questions 05 leaves open, the ones you actually hit
the moment you try to write `expected_tensors`:

1. **What is each tensor?** What does `q_proj` *do*? Why are there *two* layernorms
   per block, plus *another* two norms inside attention? What's `lm_head`?
2. **Where does this knowledge come from?** How do we know a Qwen3 block has exactly
   these eleven tensors, with these names, in this arrangement?

The second question is the important one, so it goes first — because the honest
answer changes how you build everything after.

---

## Part 1 — Where the architecture comes from (provenance)

### It is **not** in your head, and it is **not** in the config

The tempting shortcut is "I know transformers, I'll just write the blocks from
memory." Two problems:

- **Memory is a lossy cache and it's already been wrong.** Building this very
  function, memory said "tied embeddings ⟹ no `lm_head.weight` in the file." The
  actual file has one (a redundant copy — see Part 3). We only caught it because we
  read the header instead of trusting recall.
- **You cannot derive an architecture from `config.json`.** This is the thesis of
  [learning 09](09-config.md): a config *parameterizes* an architecture whose
  structure lives in **code**. Nothing in `config.json` says "apply an RMSNorm to
  the queries before scoring." That's a *design decision*, not a dial. The config
  gives you `head_dim: 128`; it cannot tell you `q_norm` exists.

So where *does* it come from? From the people who designed the model, communicated
down a **chain of trust**, each link verified against the previous by reproducing
its **numbers**:

```
  Qwen team (Alibaba)          designs the architecture, trains it, writes it down
      │   releases: technical report (arXiv:2505.09388) + reference code + weights
      ▼
  HuggingFace `transformers`   implements modeling_qwen3.py from the report + code
      │   verifies: its forward pass reproduces the creators' outputs   ← golden vectors
      ▼
  Failed Star (us)             reads HF as spec, runs HF as oracle
          verifies: our Rust logits match HF's, to tolerance            ← golden vectors
```

The design choices — QK-norm, grouped-query attention, SwiGLU — were *invented* by
the model's creators and are *documented*. Everyone downstream (including us)
implements a documented design and confirms it by getting the same numbers. That
"same numbers" test is the golden-vector methodology this whole project runs on,
applied at every link.

**The evidence is in the file's own header.** `modeling_qwen3.py` opens with:

> `Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.`

The creator (Qwen/Alibaba) and the reference library (HuggingFace) are named as
**co-authors** of the reference code. The chain of trust isn't a metaphor — it's a
copyright line. (In fact `modeling_qwen3.py` is auto-generated from
[`modular_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modular_qwen3.py);
the model team often contributes that source directly.)

### What role does `transformers` play for us? Three, kept distinct

1. **Spec — we *read* it.** To learn what tensors exist and how they're wired
   (e.g. that `q_norm` is applied to `q_proj`'s output before attention). The tensor
   *names* are literally the PyTorch attribute paths: `self.self_attn.q_proj =
   nn.Linear(...)` → the weight `model.layers.0.self_attn.q_proj.weight`.
2. **Oracle — we'll *run* it (at M2).** To dump golden logit vectors and assert our
   Rust output matches to tolerance. Same as the M0 tokenizer golden.
3. **Not copying.** We do **not** translate its Python line-by-line into Rust.

> **We copy the *architecture*, never the *code*.** We have no choice about the
> architecture: the trained weights are only meaningful under Qwen3's exact
> computation, so our math must match. But *how we express* that computation (tight
> Rust + MSL, our style) is ours, and we prove equivalence with **numbers**, not by
> diffing source. It's all **PyTorch**, by the way — which the file itself told us:
> `__metadata__: {"format": "pt"}` in the safetensors header means the weights were
> saved from a PyTorch model.

### The three sources of truth, in order of authority

| Source | Answers | For `fs inspect` |
|---|---|---|
| **The file header** | *what tensors exist* + their shapes | ground truth we cross-check against |
| **The reference code** | *how each tensor is used* (the forward pass) | how we know what to implement (M2) |
| **The config** | *the dims* that size everything | [learning 09](09-config.md) |

When we build `expected_tensors`, we derive the expected set from the **config**
(source 3), then diff it against the **header** (source 1). Any disagreement is a
loud failure at load time — not quiet garbage in M2.

---

## Part 2 — The anatomy, tensor by tensor

Straight from the real header (`python` over `model.safetensors`): **311 tensors =
3 global + 11 per block × 28 layers.** Here's what each one *is*.

### The 3 global tensors

```
model.embed_tokens.weight   [V, H] = [151936, 1024]   the token table
model.norm.weight           [H]    = [1024]            final RMSNorm scale
lm_head.weight              [V, H] = [151936, 1024]    output projection (tied — Part 3)
```

- **`embed_tokens`** — the vocabulary table. A token id is a *row index*: look up
  row `id`, get its `H`-wide vector. Not a matmul, a gather. (See
  [learning 04](04-embedding-models.md).)
- **`model.norm`** — one final RMSNorm on the output stream before the head.
- **`lm_head`** — the *output* projection: turns the final `H`-wide vector into `V`
  logits, one score per vocabulary token. "Tied" to `embed_tokens` — Part 3.

### The 11 tensors of one block (× 28)

A block is **two sub-layers** — attention, then a feed-forward MLP — each wrapped as
`norm → sub-layer → add back to the residual stream`. (The "add back" is the
residual connection; the `H`-wide stream is the bus from [learning 05](05-reading-shapes.md).)

```
── attention half ─────────────────────────────────────────────────────────
input_layernorm.weight            [H]           norm the bus before attention
self_attn.q_proj.weight           [2048, 1024]  H ─▶ 16 query heads × 128
self_attn.k_proj.weight           [1024, 1024]  H ─▶  8 kv   heads × 128
self_attn.v_proj.weight           [1024, 1024]  H ─▶  8 kv   heads × 128
self_attn.q_norm.weight           [128]         RMSNorm each query head (QK-norm)
self_attn.k_norm.weight           [128]         RMSNorm each key   head (QK-norm)
self_attn.o_proj.weight           [1024, 2048]  concat heads ─▶ H (back on the bus)
── MLP half ───────────────────────────────────────────────────────────────
post_attention_layernorm.weight   [H]           norm the bus before the MLP
mlp.gate_proj.weight              [3072, 1024]  H ─▶ I  ┐ SwiGLU:
mlp.up_proj.weight                [3072, 1024]  H ─▶ I  │ down( SiLU(gate)·up )
mlp.down_proj.weight              [1024, 3072]  I ─▶ H  ┘ then back on the bus
```

**Norms (`input_layernorm`, `post_attention_layernorm`, `model.norm`).** All three
are 1-D `[H]` *scale vectors*, not matrices. RMSNorm divides a vector by its own
root-mean-square (keeping activation magnitudes stable so nothing explodes or
vanishes as it flows through 28 layers), then multiplies element-wise by these
learned per-channel weights. Placed *before* each sub-layer ("pre-norm").

**Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).** Attention lets
each token gather information from other tokens. Three questions, three
projections:
- `q_proj` — the **query**: "what am I looking for?" (16 heads × 128 = 2048 wide)
- `k_proj` — the **key**: "what do I offer as a match?" (8 kv heads × 128 = 1024)
- `v_proj` — the **value**: "what do I carry if matched?" (8 kv heads × 128 = 1024)

The mechanism: each query is scored against every key (a dot product), the scores
are softmaxed into weights, and the values are combined by those weights. Then
`o_proj` projects the concatenated per-head results back onto the `H`-wide bus. The
query/kv asymmetry (16 vs 8) is **GQA** — see [learning 05](05-reading-shapes.md).

**QK-norm (`q_norm`, `k_norm`) — the one you can't guess.** These are `[128]`
RMSNorms (width `d`, one head) applied to *each query and key vector* right before
scoring. They keep attention logits from growing too large. **This is the poster
child for "read the reference, don't reason from first principles":** GPT-2 and
Llama-1 have no such tensors; nothing about attention *requires* them; you only know
they exist because the reference code has

```python
self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)   # and k_norm
...
query_states = self.q_norm(self.q_proj(hidden_states)...)            # applied before attention
```

(confirmed in `modeling_qwen3.py`) — and because the header lists `q_norm.weight` /
`k_norm.weight`. No config field announces them.

**MLP (`gate_proj`, `up_proj`, `down_proj`) — SwiGLU, which is why there are two
"up" projections.** A vanilla FFN is `down(act(up(x)))` — one projection up, one
back. SwiGLU splits the "up" into two:

```
   MLP(x) = down_proj( SiLU(gate_proj(x)) · up_proj(x) )
                        └── the gate ──┘   └─ the value ─┘   (· = element-wise)
```

`gate_proj` and `up_proj` both take `H ─▶ I` (3072 wide); the gate is squashed by
the SiLU activation and *multiplies* the up-projection element-wise (a "gated" unit
— the model learns to attenuate parts of the value). `down_proj` brings `I ─▶ H`,
back onto the bus. Two `[I,H]` tensors + one `[H,I]` is the SwiGLU signature; a
single up-projection would be a plain FFN.

---

## Part 3 — The `lm_head` surprise (a worked example of "read, don't assume")

`config.json` has `tie_word_embeddings: true`. **"Tied" means the output projection
*is* the embedding table — the same weights, used twice** (row lookup on the way in;
transposed as the `H ─▶ V` output projection on the way out). It is a statement
about the **math**, not a promise about the **file**.

Memory said "tied ⟹ the file omits `lm_head.weight`." The header says otherwise:

```
lm_head.weight               [151936, 1024]   ← present!
model.embed_tokens.weight    [151936, 1024]
```

and their raw bytes are **byte-for-byte identical** (verified). Qwen3-0.6B is tied
*and* ships a redundant copy of the table. The consequences:

| | count |
|---|---:|
| stored tensors | **311** |
| stored params (naive sum) | **751,632,384** (~751M) |
| logical params (dedup the tied copy) | **596,049,920** (~596M) — this is the "0.6B" |
| embedding share of logical | **26.1%** |

A quarter of the model is the vocabulary table — stored **twice**. This is why
`expected_tensors` marks `lm_head.weight` **optional when tied** (a tied file may
omit it *or* duplicate it — both legal), and why `cross_check` reports
`stored_params` *and* `logical_params` separately, flagging the redundant copy as a
*note*, not a failure. Had we coded from the memory-assumption, the cross-check would
have called a real tensor an "extra" and reported a false failure on the real model.

**The lesson, one line:** derive the expectation from the config, but confirm it
against the header and the reference — never against recall.

`fs inspect models/qwen3-0.6b` now prints exactly this reconciliation — the deduped
`596M`, the redundant-copy note, the `26.1%` embedding share — all *derived* from
`config.json` and checked against the file, none of it hard-coded. See the abridged
output in [learning 05](05-reading-shapes.md#how-this-shows-up-in-fs-inspect-and-in-the-code).

---

## Cross-links

- ⬅ [`learning 05 · reading shapes`](05-reading-shapes.md) — the *shapes* of these
  same tensors (the `[out,in]` rule, GQA, the tied-embeddings param math).
- ⬅ [`learning 09 · config.json`](09-config.md) — why a config *parameterizes* an
  architecture whose structure lives in code (the root of Part 1).
- ⬅ [`learning 04 · embedding models`](04-embedding-models.md) — `embed_tokens` as a
  gather, and the three senses of "embedding."
- ➡ `docs/02-weights.md` — the M1 milestone writeup (uses this anatomy).
- 🔧 [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
  — the reference forward pass (spec + M2 oracle).
- 📄 [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
  — the creators' design rationale.
- 🧭 [Raschka · Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
  — QK-norm / GQA / SwiGLU across models.
