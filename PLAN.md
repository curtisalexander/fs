# PLAN — the Failed Star milestone curriculum

A slow, multi-session build. Each milestone is **a runnable artifact** plus **a
doc**. We go in order; each builds on the last. Read [`docs/00-map.md`](docs/00-map.md)
first for the big picture, and [`PROGRESS.md`](PROGRESS.md) each session for "where
are we."

**Method per milestone** (Raschka-style, see map §6): concept (book) → config (HF)
→ reference (`ds4`) → build (Rust/MSL) → verify (golden vector) → document.

**Done-check philosophy:** a milestone is done when it *runs* and its output is
*verified* against a known-good reference, and its doc cross-links book + ds4.

Legend: ☐ todo · ◐ in progress · ☑ done

---

## Decisions locked (the "spirit" — keep these honest)

- **Host = Rust. Kernels = MSL.** Metal via **raw FFI / ObjC runtime, no wrapper
  crate.** Keep it *tight* — readable, fast, low memory, like `ds4`.
- **First model = Qwen3-0.6B** (tiny dense): GQA + RoPE + SwiGLU + RMSNorm.
- **Correctness = golden vectors** from the official implementation. Python only
  ever appears as a one-shot oracle, never as a second engine.
- **Scope = Metal/macOS only.** No CUDA/ROCm, no server/agent/distributed (those
  stay as things to *read* in `ds4`, not build).
- **Two products:** the `fs` engine *and* the cross-linked docs that teach it.

---

## M0 — Tokenizer  ☑  *(done — see [`docs/01-tokenizer.md`](docs/01-tokenizer.md))*
Text ↔ token IDs. BPE encode/decode against the chosen model's real vocabulary.
- **Artifact:** `fs tokenize "hello world"` → IDs, and decode back to text, in Rust.
- **Verify:** round-trip + match the official tokenizer's IDs on a set of strings.
- 📖 §2.2 (p.46) · 🔧 `reference/ds4/ds4.c` (BPE + `str_i32_table` hash table) · 🧭 Raschka "LLM from scratch" BPE.
- **Why first:** no GPU, no weights, self-contained; it's the model's front door.

## M1 — Load the weights  ◐  *(current)*
Parse the model file format and map every tensor (names, shapes, dtypes) into
memory. Read `config.json` (layers, dims, heads, vocab).
- **Artifact:** `fs inspect model/` prints the architecture + tensor table.
- **Verify:** shapes/counts match the HF config; checksum a few tensors.
- 📖 §4.2.2 "Model File Formats" (p.103) · 🔧 `ds4` GGUF path (`ds4.c` "owns GGUF
  loading", mmap-based) + `gguf-tools/`.
- **Format decision (proposed):** **safetensors for M1–M4** (Qwen3-0.6B ships it
  natively; trivial to parse; clean bf16 for correctness), then **GGUF at M5**
  alongside ds4's parser when quantization is the lesson. ds4 itself is GGUF-only.

## M2 — Forward pass → logits  ☐  *(the "it understands" milestone)*
Embeddings → N transformer blocks (RMSNorm, RoPE, attention, SwiGLU) → final norm
→ logits. **CPU/Rust first** (slow but clear), correctness over speed.
- **Artifact:** `fs logits "The capital of France is"` prints top-k next tokens.
- **Verify:** logits match the official implementation's golden vector (tight tol).
- 📖 §2.1 (p.42), §2.2.2 (p.50), §2.2.3 (p.52) · 🔧 `ds4.c` + `metal/{norm,dsv4_rope,flash_attn,glu,dense,get_rows}.metal`.
- **Sub-steps:** matmul → RMSNorm → embedding gather → RoPE → attention (one head,
  then GQA) → SwiGLU FFN → stack the block → full model.

## M3 — Sampling → generation  ☐  *(the "it's alive" milestone)*
Softmax+temperature, greedy/top-k/top-p, the autoregressive loop, streaming output,
stop tokens.
- **Artifact:** `fs chat "..."` streams a (slow) reply to the terminal.
- **Verify:** greedy decode reproduces the reference's greedy continuation.
- 📖 §2.2 (p.46) sampling · 🔧 `metal/{softmax,argsort}.metal`.

## M4 — KV cache  ☐  *(the "I made it faster" milestone)*
Cache K/V per layer; decode does one-token work. RAM-only first.
- **Artifact:** decode tok/s jumps; correctness unchanged vs M3.
- **Verify:** same output as M3, measurably faster; benchmark prefill vs decode.
- 📖 §5.3 (p.136) · 🔧 `ds4_kvstore.c/.h`, `metal/dsv4_kv.metal` (SSD streaming = read-only study).

## M5 — Quantization  ☐  *(the "it gets small" milestone)*
Load/dequant 8-bit then 4-bit weights; measure quality + speed + memory.
- **Artifact:** `fs` runs a quantized model; memory drops, decode speeds up.
- **Verify:** perplexity/quality delta vs fp16 within a documented budget.
- 📖 §5.1 (p.120) · 🔧 `gguf-tools/`, `gguf-tools/imatrix/`, dequant in `metal/*`.

## M6 — Metal acceleration  ☐  *(the "feel the hardware" milestone)*
Port the hot ops to MSL kernels, driven from Rust via **raw Metal FFI**. Get the
GPU doing the matmuls/attention. Then **kernel fusion**.
- **Artifact:** decode tok/s on the M5 GPU; a real interactive speed.
- **Verify:** GPU output matches CPU output; benchmark each fused kernel.
- 📖 §4.1 (p.96) incl. fusion (p.100), §3.1 (p.74), §3.5 (p.89) · 🔧 `ds4_metal.m` + all of `metal/`.
- **Note:** the FFI scaffolding (device/queue/buffers/pipelines) is its own sub-task.

## M7+ — Stretch goals  ☐
Pick by interest once the core engine breathes:
- **Speculative decoding** (§5.2, p.129) — draft/target.
- **MoE** (§2.2.4, p.53) — routing + expert FFNs (`metal/moe.metal`).
- **DeepSeek-style compressed attention / MLA** (`dsv4_hc.metal`, `dsv4_kv.metal`;
  Raschka MLA notes) — the leap toward `ds4`'s actual model.
- **On-disk KV / SSD streaming** (`ds4_ssd.c`) — `ds4`'s signature idea.
- A tiny **server** or **REPL** for ergonomics.

---

## Parallel track — the docs (always-on)
After each milestone, write the next numbered doc after the map — e.g.
`docs/01-tokenizer.md` for M0, `docs/02-weights.md` for M1,
`docs/03-forward-pass.md` for M2: what we built, the math, the gotchas, and the
three-way cross-links (book §/page, `ds4` file, Raschka). Start in Markdown;
graduate the best ones to **rich HTML with diagrams** once content settles. Index
everything in [`docs/RESOURCES.md`](docs/RESOURCES.md).

**Learnings get their own home on the site.** The `docs/learnings/` notes are the
Markdown source of truth *and* graduate into a dedicated **Learnings** section on
the HTML site (its own nav entry + index), hand-distilled like the rest, linked
from the doc/milestone that references them (link the `.html`, not the raw `.md`).
HTML is where learnings earn nicer diagrams and *sparing* interactivity (à la
`diagrams.html`). See [`docs/dev-loop.md`](docs/dev-loop.md) → "Learnings → the
site's Learnings section" for the ritual. *(Back-graduation of `learnings/01–04`
is owed — see `PROGRESS.md`.)*
