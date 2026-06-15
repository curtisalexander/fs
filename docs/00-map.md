# 00 — The Map: how an LLM inference engine works (and how we'll build one)

This is the big picture. Read it once to see the whole machine, then come back to
it as an index: every part links to **(a)** the relevant section of *Inference
Engineering* (the book), **(b)** the file in `ds4` that implements it for real,
and **(c)** Raschka's writing for architecture context.

> **New here?** Read [`prerequisites.md`](prerequisites.md) first — what to know
> before diving in (and what we'll demonstrate). It also expands this map's ladder
> into "which idea each rung needs."

> **How to use this doc.** It's organized as an **abstraction ladder** — from the
> highest level (a chat loop) down to the lowest (threads on the GPU). You can
> stop descending at whatever rung stops being interesting *to you*. Each rung
> says what it is, why it matters, and where to look in all three sources.

---

## 0. The one-sentence version

**Inference is: turn text into numbers, push those numbers through a big pile of
matrix multiplications that the model learned during training, and turn the
resulting numbers back into text — one token at a time, in a loop.**

Everything else — KV caches, quantization, Metal kernels, MoE routing, SSD
streaming — is *making that loop correct, then fast, then small enough to fit.*

📖 Book: **Ch 0 "Inference" (p.15)**, **Ch 2 "Models" (p.39)**.

---

## 1. The abstraction ladder (the whole machine at a glance)

```
┌───────────────────────────────────────────────────────────────────────┐
│ RUNG 7  Chat / agent loop      "user types, model replies, repeat"    │  highest
│         prompt templating, tool calls, sessions                       │
├───────────────────────────────────────────────────────────────────────┤
│ RUNG 6  Generation loop        prefill → decode → sample → append     │
│         temperature, top-k/top-p, stop tokens, streaming              │
├───────────────────────────────────────────────────────────────────────┤
│ RUNG 5  The model forward pass  embeddings → N transformer blocks →   │
│         final norm → logits                                           │
├───────────────────────────────────────────────────────────────────────┤
│ RUNG 4  One transformer block   norm → attention (+RoPE, +KV cache)   │
│         → norm → FFN/MoE → residual adds                              │
├───────────────────────────────────────────────────────────────────────┤
│ RUNG 3  Tensor operations       matmul, softmax, RMSNorm, RoPE,       │
│         SwiGLU, argmax/sampling                                       │
├───────────────────────────────────────────────────────────────────────┤
│ RUNG 2  Kernels                 each op as a Metal (MSL) shader, run  │
│         on the GPU over a command buffer                              │
├───────────────────────────────────────────────────────────────────────┤
│ RUNG 1  Memory & numbers        weights in RAM, dtype/quantization,   │
│         unified memory, buffers, bandwidth                            │
├───────────────────────────────────────────────────────────────────────┤
│ RUNG 0  The hardware            Apple Silicon GPU: threads, SIMD,     │  lowest
│         threadgroups, registers, the memory hierarchy                 │
└───────────────────────────────────────────────────────────────────────┘
```

We will *build* roughly **bottom-meets-middle**: enough of rungs 1–3 to make
rung 5 (a forward pass) produce correct logits, then rung 6 to make it *talk*,
then descend into rung 2/1/0 to make it *fast*.

---

## 2. The data's journey (follow one prompt through the machine)

This is the same story as the ladder, but told as a pipeline. Each stage names
the milestone (`Mx`, see [`../PLAN.md`](../PLAN.md)) where we build it.

### Stage A — Text → tokens *(the tokenizer)* · **M0**
Text is split into integer **token IDs** using a learned vocabulary (BPE). `"hello
world"` → `[15339, 1917]`. At the end we do the reverse to print output.

- 📖 Book: tokens are introduced in **§2.2 "LLM Inference Mechanics" (p.46)**.
- 🔧 `ds4`: vocabulary/tokenizer machinery uses a **radix tree** — `reference/ds4/rax.c`,
  `rax.h` (antirez's `rax` library). Tokenizer logic lives in `ds4.c`.
- 🧭 Raschka: tokenization is a prerequisite he assumes; see his "Build a Large
  Language Model (From Scratch)" for a clean BPE walkthrough.

### Stage B — Tokens → vectors *(the embedding table)* · **M2**
Each token ID indexes a row of the **embedding matrix**, turning each token into a
vector of `d_model` floats. Now the model works on numbers, not text.

- 📖 Book: **§2.1 "Neural Networks" (p.42)**, **§2.1.1 "Linear Layers and Matmul" (p.44)**.
- 🔧 `ds4`: row gather is the `get_rows` kernel — `reference/ds4/metal/get_rows.metal`.

### Stage C — The stack of transformer blocks *(the model)* · **M2**
The vectors pass through **N identical transformer blocks**. Each block has two
sub-layers, each wrapped in a normalization and a residual connection:

```
x → RMSNorm → Attention → (+x)  →  RMSNorm → FFN/MoE → (+)  → x'
```

**Attention** lets each token mix in information from earlier tokens. **FFN**
(feed-forward, often SwiGLU) is a per-token transformation. Stacked N times, this
is the whole model body.

- 📖 Book: **§2.2.2 "Transformer Blocks" (p.50)**, **§2.2.3 "Attention" (p.52)**,
  **§2.2.4 "Mixture of Experts Models" (p.53)**.
- 🔧 `ds4`: the block is orchestrated in `ds4.c`; the heavy pieces are individual
  shaders — `metal/flash_attn.metal` (attention), `metal/norm.metal` (RMSNorm),
  `metal/dsv4_rope.metal` (RoPE), `metal/glu.metal` (SwiGLU/GLU), `metal/moe.metal`
  (mixture-of-experts routing), `metal/dense.metal` (matmuls).
- 🧭 Raschka: the
  [architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
  is a tour of exactly these choices across modern models (MHA vs GQA vs MLA,
  pre/post-norm, RoPE/NoPE, SwiGLU, MoE shared-expert designs).

Components inside a block, each its own tensor op (**RUNG 3**):

| Component | What it does | 📖 Book | 🔧 ds4 shader |
|---|---|---|---|
| **RMSNorm** | stabilizes activations before each sub-layer | §2.1 (p.42) | `metal/norm.metal` |
| **RoPE** | injects token *position* by rotating Q/K | §2.2.3 (p.52) | `metal/dsv4_rope.metal` |
| **Attention** | each token attends to past tokens (Q·Kᵀ → softmax → ·V) | §2.2.3 (p.52), §2.5 (p.67) | `metal/flash_attn.metal`, `metal/softmax.metal` |
| **GQA** | share K/V across query heads to shrink the KV cache | §2.2.3, §2.5 | (in attention path) |
| **FFN / SwiGLU** | per-token MLP with a gating nonlinearity | §2.1.2 (p.44) | `metal/glu.metal`, `metal/dense.metal` |
| **MoE** | route each token to a few of many expert FFNs | §2.2.4 (p.53) | `metal/moe.metal` |

### Stage D — Final norm → logits *(the LM head)* · **M2**
A last RMSNorm, then a big matmul against the vocabulary projection produces a
**logit** (a score) for every token in the vocabulary — for the *next* token.

- 📖 Book: **§2.2.1 "LLM Architecture" (p.49)**.
- 🔧 `ds4`: final matmul via `metal/dense.metal`.

### Stage E — Logits → next token *(sampling)* · **M3**
Convert logits to probabilities (softmax with **temperature**), then pick a token:
greedy (argmax), or sample with **top-k / top-p**. This single chosen token is the
model's output for this step.

- 📖 Book: sampling is part of **§2.2 (p.46)**; quality/latency tradeoffs recur in Ch 5.
- 🔧 `ds4`: `metal/softmax.metal`, `metal/argsort.metal` (top-k/sorting for sampling).

### Stage F — Append and repeat *(the autoregressive loop)* · **M3 → M4**
Append the new token to the sequence and go back to Stage C — but only for the
*one* new token, because of the **KV cache** (below). Loop until a stop token or
length limit. Stream tokens to the screen as they're produced.

- 📖 Book: **Ch 5 "Techniques" (p.117)**, esp **§5.3 "Caching" (p.136)**.

---

## 3. The two phases: prefill vs decode (why inference feels the way it does)

The loop has two very different regimes — internalizing this explains almost every
performance decision in the book and in `ds4`:

- **Prefill** — process the *whole prompt* at once. Many tokens → big matmuls →
  **compute-bound**. This is fast per-token; it's the "reading your question" phase.
- **Decode** — generate one token at a time. Each step touches *all the weights*
  to produce *one* token → **memory-bandwidth-bound**. This is the slow,
  one-word-at-a-time "typing the answer" phase.

The headline `ds4` numbers show this split directly (M5 Max, q2): **~463 tok/s
prefill vs ~26 tok/s decode**. Same model, ~18× difference — because decode is
gated by how fast you can stream weights out of memory, not by math.

- 📖 Book: **§2.4 "Calculating Inference Bottlenecks" (p.61)** — esp **§2.4.1
  "Ops:Byte Ratio and Arithmetic Intensity" (p.62)** and **§2.4.2 "LLM Inference
  Bottlenecks" (p.63)**. This is the single most clarifying section for *why* the
  optimizations exist.

---

## 4. The three things that make it practical (correct → fast → small)

Once the loop is *correct*, almost all of inference engineering is these three:

### KV cache — don't recompute the past · **M4**
Attention at step *t* needs the keys/values of all earlier tokens. Recomputing
them every step is quadratic waste. So we **cache** K and V per layer and only
compute the new token's K/V each step. This is what turns decode from "reprocess
everything" into "one token's worth of work."

- 📖 **§5.3 "Caching" (p.136)** → §5.3.1 prefix caching/reuse (p.136), §5.3.2
  *where* to store it (p.139), §5.3.4 long context (p.141).
- 🔧 `ds4`: `ds4_kvstore.c/.h`, and `metal/dsv4_kv.metal`. **`ds4`'s big idea:**
  *"the KV cache is a first-class disk citizen"* — it streams KV from SSD
  (`ds4_ssd.c`), turning "fits in RAM?" from a yes/no into a speed dial.

### Quantization — store weights in fewer bits · **M5**
Weights are trained in 16-bit but can be stored in 8/4/2-bit with clever scaling,
shrinking memory and (because decode is bandwidth-bound) *speeding it up*. This is
how a model fits 64GB at all.

- 📖 **§5.1 "Quantization" (p.120)** → §5.1.1 number formats (p.121), §5.1.2
  approaches (p.125), §5.1.3 measuring quality impact (p.128).
- 🔧 `ds4`: quant tooling in `reference/ds4/gguf-tools/` (incl. `imatrix/` for
  calibration); uses IQ2_XXS for routed experts, Q2_K for down-projections;
  dequant kernels in the `metal/` shaders. Number-format note: this is **RUNG 1**.

### Metal kernels — do the math on the GPU, tightly · **M6**
Every tensor op (RUNG 3) becomes a **kernel** (RUNG 2): a small MSL program the
GPU runs over thousands of threads. We submit them via the Objective-C Metal API
**through raw FFI** (no wrapper crate) so the buffer/command/pipeline machinery
stays visible. **Kernel fusion** (doing several ops in one kernel to avoid
round-tripping memory) is a key speed lever.

- 📖 **§4.1 "CUDA" (p.96)** — the concepts (kernels p.98, **kernel fusion p.100**)
  transfer directly to Metal; **§3.5 "Local Inference" (p.89)** and **§3.1 "GPU
  Architecture" (p.74)** for the hardware (RUNG 0).
- 🔧 `ds4`: the entire `reference/ds4/metal/` directory (19 shaders) + the Metal
  host backend `ds4_metal.m` (~26k lines — this is most of how `ds4` runs on a Mac).

### …and the advanced stuff (later) · **M7+**
**Speculative decoding** (§5.2, p.129 — a small draft model proposes tokens a big
model verifies), **MoE** at scale (§2.2.4), **model parallelism** (§5.4, p.142),
**DeepSeek-V4's compressed attention** (`ds4`'s `dsv4_hc.metal` / `dsv4_kv.metal`;
Raschka's MLA notes). These are stretch goals once the core engine breathes.

---

## 5. Failed Star vs Dwarf Star vs the Book — who covers what

| Layer | Book teaches | `ds4` implements | Failed Star will build |
|---|---|---|---|
| Concepts/vocabulary | ✅ everything | — | — (we cite the book) |
| Tokenizer | mentions | `rax.c` + `ds4.c` | M0 (Rust) |
| Forward pass | ✅ the math | `ds4.c` + `metal/*` | M2 (Rust + MSL) |
| Sampling | ✅ | `metal/softmax,argsort` | M3 (Rust) |
| KV cache | ✅ §5.3 | `ds4_kvstore.c`, SSD streaming | M4 (Rust, RAM-only first) |
| Quantization | ✅ §5.1 | `gguf-tools/`, 2-bit experts | M5 (start 8/4-bit) |
| Metal kernels | ✅ §4.1 (as CUDA) | `ds4_metal.m` + `metal/*` | M6 (raw FFI + MSL) |
| Multi-backend (CUDA/ROCm) | Ch 3–4 | `ds4_cuda.cu`, `ds4_rocm.cu` | ❌ out of scope (Metal only) |
| Distributed / server / agent | Ch 7 | `ds4_distributed.c`, `ds4_server.c`, `ds4_agent.c` | ❌ out of scope (for now) |

**The model gap to remember:** `ds4` runs **DeepSeek-V4-Flash** (284B total / 13B
active **MoE**, 1M context, exotic **compressed** attention). Failed Star starts
with a **tiny dense** model (vanilla GQA + RoPE + SwiGLU). So the comparison is
**1:1 for fundamentals** (RoPE, RMSNorm, matmul, softmax, KV-cache concept, kernel
structure) and **divergent for the fancy parts** (MoE, compressed attention) —
which is exactly why those are *late* milestones.

---

## 6. Our working method (adapted from Raschka's workflow)

Raschka's loop for understanding a model: **read the technical report → inspect the
HF config files → read the reference implementation ("working code doesn't lie") →
implement a few pieces by hand.** We apply that *per milestone*:

1. **Concept** — read the book section(s) for this milestone.
2. **Config** — look at the real model's `config.json` / weights on Hugging Face.
3. **Reference** — read how `ds4` does it (and Raschka's notes for architecture).
4. **Build** — implement it ourselves in Rust (+ MSL).
5. **Verify** — match a **golden vector** (logits/activations from the official
   implementation). This is `ds4`'s *"official-vector validation"* in miniature.
6. **Document** — write the milestone's doc, cross-linking all three sources.

---

## 7. What's next

➡️ **M0 — The Tokenizer.** Text ↔ token IDs in Rust. Self-contained, no GPU, no
model weights, immediately runnable — the gentlest possible on-ramp, and the first
and last thing every prompt touches. See [`../PLAN.md`](../PLAN.md).

For the full cross-reference index (book page numbers, every `ds4` file, Raschka
links), see [`RESOURCES.md`](RESOURCES.md).
