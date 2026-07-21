# Learning 09 — `config.json`: the model's blueprint (and why a dozen numbers suffice)

> **Date:** 2026-07-14 · **Context:** M1, reading `config.json` into `Config` · **Status:** living
>
> 📖 *Inference Engineering* §4.2 "Model File Formats" (p.103) — config alongside the weights
> 🔧 `ds4`: GGUF has **no separate config** — the hyperparameters live as metadata
> key/values *inside* the `.gguf` file (`ds4.c` reads them there)
> 🧭 Hugging Face `PretrainedConfig` + the `architectures` → `AutoModel` registry

We just wrote [`Config::load`](../../src/config.rs). It reads a ~1 KB text file and
hands back thirteen numbers. That's a strange thing to sit next to a **1.4 GB** pile
of weights and call an equal partner — but it is one. This note is what that little
file *is*, why you can't run the model without it, and the question it should
provoke: **how can a config file possibly describe a whole neural network?**

The honest answer — the one worth the whole note — is: **it can't, and it doesn't.**
It parameterizes one. Hold that thought; we'll earn it below.

---

## A model is (at least) two files, and they're useless apart

```
  config.json            model.safetensors
  ┌───────────────┐      ┌──────────────────────────┐
  │ the blueprint │      │ the material              │
  │ ~1 KB of text │      │ ~1.4 GB of raw bf16 bytes │
  │ 13 numbers    │      │ 596M weights, no labels   │
  └───────┬───────┘      └────────────┬─────────────┘
          │                           │
          └──────────── need BOTH ────┘
                         │
                    a runnable network
```

- **`model.safetensors` alone** is a flat blob. Nothing in it says a `[2048, 1024]`
  matrix is `q_proj`, or that there are 28 blocks, or where one layer ends and the
  next begins. It's numbers without grammar. (See
  [`learning 01`](01-safetensors-vs-gguf.md) — "the bytes *are* the numbers," but
  the bytes don't know what they're *for*.)
- **`config.json` alone** is a recipe with no ingredients: it says "28 layers, hidden
  width 1024, 16 query heads over 8 KV heads" — the *shape* of a network with no
  learned values in it.

You need **both**: the config to know how to fold the blob into a network, the blob
to fill that network with what it learned. Our whole M1 `fs inspect` is exactly this
handshake — it proves the tensors in the `.safetensors` file are the ones the config
*implies*, and that their shapes line up. A checksum can't do that; only the config
can say what "correct" even means.

> This "need both" point belongs in the M1 milestone writeup too
> (`docs/m1-weights.md`, owed at M1 close) — state it from the weights side there,
> from the config side here.

---

## What's actually in Qwen3-0.6B's `config.json`

Twenty-six keys. They sort into three piles, and the *sorting itself* is the lesson.

### Pile A — the dimensions we build every shape from (7)

These are the [`learning 05`](05-reading-shapes.md) legend; we don't re-derive them
here, just point at them:

| field | value | symbol |
|-------|------:|:------:|
| `vocab_size` | 151936 | `V` |
| `hidden_size` | 1024 | `H` |
| `num_hidden_layers` | 28 | `L` |
| `head_dim` | 128 | `d` |
| `num_attention_heads` | 16 | — |
| `num_key_value_heads` | 8 | — |
| `intermediate_size` | 3072 | `I` |

Every weight matrix's shape is a product of these. Get one wrong and `fs inspect`
lights up.

### Pile B — scalars we parse and spend later (6)

| field | value | where it's used |
|-------|------:|-----------------|
| `rms_norm_eps` | 1e-6 | RMSNorm's divide-by-zero guard (M2) |
| `rope_theta` | 1000000 | RoPE base frequency (M2) |
| `tie_word_embeddings` | true | reuse the embedding matrix as `lm_head` — **changes the expected tensor set** (M1) |
| `bos_token_id` | 151643 | begin-of-sequence marker (M3) |
| `eos_token_id` | 151645 | stop token (M3) |
| `max_position_embeddings` | 40960 | context-length ceiling |

### Pile C — fields we *ignore*, and exactly why that's safe

This is the interesting pile. A field is safe to ignore for one of two reasons —
**it's training-only**, or **its value happens to match what we hardcoded**:

| field | value | why we can skip it |
|-------|------:|--------------------|
| `architectures` | `["Qwen3ForCausalLM"]` | names the **code class** to build. *We are that class* — see below. |
| `model_type` | `"qwen3"` | HF registry key that resolves to the same class. |
| `hidden_act` | `"silu"` | activation. We **hardcode SwiGLU** in M2; if this said `"gelu"` our engine would be *wrong*, not reconfigurable. |
| `torch_dtype` | `"bfloat16"` | a hint. We read each tensor's dtype from the safetensors header instead — authoritative, per-tensor. |
| `attention_bias` | false | q/k/v/o have **no bias**; matches our bias-free `Linear`. If true we'd need to load bias tensors. |
| `attention_dropout` | 0.0 | training-only; inference never drops. |
| `initializer_range` | 0.02 | training-only (weight init). |
| `rope_scaling` | null | no long-context frequency scaling; if set we'd bend RoPE. |
| `sliding_window` / `use_sliding_window` / `max_window_layers` | null / false / 28 | sliding-window attention is **off**; full causal attention is correct. |
| `use_cache` | true | KV-cache is our M4 decision regardless. |
| `transformers_version` | `"4.51.0"` | which library wrote the file. Pure metadata. |

> **A quiet trap worth naming.** Several Pile-C fields are safe *only because of
> their value*. `attention_bias: false`, `rope_scaling: null`,
> `sliding_window: null` — flip any of those and the architecture silently changes
> under us while our code keeps computing the old one. The fail-loud move (a good
> future hardening of `Config::load`) is to **assert** these are the values we
> assume, not to skip them. Ignoring a field and *asserting* a field look identical
> until the day someone hands you a model where it differs.

---

## The real question: how can ~13 numbers describe a network?

Because **they don't describe the network — they configure code that already knows
the network.** Look again at the field that does the heavy lifting:

```json
"architectures": ["Qwen3ForCausalLM"]
```

That's not data. It's a **pointer to a class** — a `~1500`-line Python file in the
`transformers` library whose body *is* the architecture: embed → (RMSNorm → RoPE →
GQA attention → residual → RMSNorm → SwiGLU → residual) × L → RMSNorm → tied
`lm_head`. The wiring, the causal mask, the order of operations, which norm, which
activation — all of that lives in **code**. `config.json` only fills in the *free
numbers* of a structure that's already fixed.

Put sharply:

> **`config.json` is not a blueprint of the network. It's the arguments to a
> constructor whose body is code.** `architectures` names the constructor; the other
> fields are its dials.

So the "two files" story is really **three things**, and one is usually invisible:

```
  architecture (CODE)  +  config (DIALS)  +  weights (LEARNED VALUES)  =  a model
  ───────────────────     ──────────────     ────────────────────────
  Qwen3ForCausalLM        config.json        model.safetensors
  the structure           the hyperparams    the parameters
```

Most people never see the code half, because everyone shares the *same* code
(`transformers`), so "a model" collapses to "config + weights." **We** don't get that
luxury — the whole point of `fs` is that **we are writing `Qwen3ForCausalLM`
ourselves.** Our engine *is* the code column. That's the deepest reason we can ignore
`hidden_act: "silu"`: we already committed to SwiGLU in our own source. The config
can't reconfigure a decision we baked into code.

### Then why does it *feel* like the config is enough?

Because the field standardized. Modern LLMs converged, hard, on **one template**: the
decoder-only transformer block, stacked `L` times. Within that template, models
differ almost entirely in two small ways:

1. **The ~7 dimension numbers** (Pile A), and
2. **A handful of "which variant" switches** — norm type, activation, positional
   scheme, attention flavor (MHA → GQA → MLA), sliding window on/off.

A dozen numbers span the *whole current family* because the family is a monoculture.
That's a fact about **this moment in the field**, not a law of neural networks.

### So is the schema "flexible enough for new architectures"?

Two cases, and they answer it precisely:

- **Variants inside the transformer family** (Mixture-of-Experts, MLA, a new RoPE
  scaling, a sliding window): expressed as **new config fields + new code branches**.
  HF configs are just JSON dicts subclassing `PretrainedConfig`, so adding a field
  (`num_experts`, `moe_intermediate_size`, `q_lora_rank`) is trivial. But the field
  is **inert until code is written to read it.** The schema grows by accretion, one
  field per capability, each backed by code.

- **Genuinely new families** (state-space models like Mamba, diffusion LMs, an RNN
  revival): a *different vocabulary entirely*. Mamba's config has `d_state`, `d_conv`,
  `expand` — and no `num_attention_heads`, because it has no attention. The JSON
  *container* is universal; the *words inside* are architecture-specific and mean
  nothing without that architecture's code.

Which lands the real answer to "is it that flexible a schema?":

> It's flexible **because it's barely a schema at all** — an open key/value dict that
> delegates every structural decision to code. That's not the config being powerful;
> it's the config being *humble*. A config can never describe an architecture nobody
> has coded yet. It's "fill in the blanks," never "describe anything."

The surprise you felt — *"how can a config explain a network?"* — is the right
instinct. It can't. It never did. The code explains the network; the config just says
how big.

---

## Mental model to keep

> **Config = the dials. Code = the wiring. Weights = the learned values.** You need
> all three to run a model; `config.json` + `model.safetensors` are only two of them,
> and they're only enough because the *third* — the architecture code — is assumed.
> On this project we're building that third thing, so we see the whole machine.
> `architectures` names it; everything else in `config.json` is just its arguments.

---

## Cross-links

- ⬅ [`learning 01 · safetensors vs GGUF`](01-safetensors-vs-gguf.md) — the "a model
  is two things" split; note GGUF folds the config *into* the weight file as metadata.
- 🔗 [`learning 05 · reading shapes`](05-reading-shapes.md) — Pile A in depth: how the
  seven dimensions build every tensor's shape (`[out,in]`, head_dim decoupling, GQA).
- ➡ `docs/m1-weights.md` (M1, owed) — the weights side of "need both"; `fs inspect`
  as the config↔weights handshake.
- ➡ `docs/03-forward-pass.md` (M2) — where Pile B's `rms_norm_eps` / `rope_theta` and
  the hardcoded SwiGLU / causal mask (the *code* half) finally run.
- 🔧 `src/config.rs` — `Config`, the typed extractors, and the fail-loud "no silent
  defaults" stance this note argues for extending to Pile C.
