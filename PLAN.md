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
- **Product scope = modern Apple Silicon + Metal.** The clear CPU reference may
  happen to run elsewhere for learning and orb host checks, but Linux portability
  is not a supported product. No CUDA/ROCm or server/agent/distributed runtime.
- **Verification boundary:** normal GitHub CI runs model-free `fmt`, build, tests,
  and clippy on a macOS Apple Silicon runner. Asset-backed model checks and
  all Metal correctness/performance checks run explicitly on the development Mac;
  an orb is useful for editing and host checks, but is never authoritative.
- **Two products:** the `fs` engine *and* the cross-linked docs that teach it.

---

## M0 — Tokenizer  ☑  *(done — see [`docs/m0-tokenizer.md`](docs/m0-tokenizer.md))*
Text ↔ token IDs. BPE encode/decode against the chosen model's real vocabulary.
- **Artifact:** `fs tokenize "hello world"` → IDs, and decode back to text, in Rust.
- **Verify:** round-trip + match the official tokenizer's IDs on a set of strings.
- 📖 §2.2 (p.46) · 🔧 `reference/ds4/ds4.c` (BPE + `str_i32_table` hash table) · 🧭 Raschka "LLM from scratch" BPE.
- **Why first:** no GPU, no weights, self-contained; it's the model's front door.

## M1 — Load the weights  ☑  *(done — see [`docs/m1-weights.md`](docs/m1-weights.md))*
Parse the model file format and map every tensor (names, shapes, dtypes) into
memory. Read `config.json` (layers, dims, heads, vocab).
- **Artifact:** `fs inspect model/` prints the architecture + tensor table.
- **Verify:** shapes/counts match the HF config; checksum a few tensors.
- 📖 §4.2.2 "Model File Formats" (p.103) · 🔧 `ds4` GGUF path (`ds4.c` "owns GGUF
  loading", mmap-based) + `gguf-tools/`.
- **Format decision:** safetensors is the native correctness path. GGUF is a
  separate, optional interoperability decision after the core engine; it is not
  coupled to quantization. `ds4` itself is GGUF-only.

## M2 — Forward pass → logits  ◐  *(current — the "it understands" milestone)*
Embeddings → N transformer blocks (RMSNorm, RoPE, attention, SwiGLU) → final norm
→ logits. **CPU/Rust first** (slow but clear), correctness over speed.
- **Artifact:** `fs logits "The capital of France is"` prints top-k next tokens.
- **Verify:** logits match the official implementation's golden vector (tight tol).
- 📖 §2.1 (p.42), §2.2.2 (p.50), §2.2.3 (p.52) · 🔧 `ds4.c` + `metal/{norm,dsv4_rope,flash_attn,glu,dense,get_rows}.metal`.
- **Sub-steps:** matmul → RMSNorm → embedding gather → RoPE → attention (one head,
  then GQA) → SwiGLU FFN → stack the block → full model.

## M3 — Deterministic generation, then sampling  ☐  *(the "it's alive" milestone)*
Build the autoregressive loop first with greedy selection and stop tokens; only
after deterministic parity add temperature/top-k/top-p sampling and streaming.
- **Artifact:** `fs generate "..."` produces a deterministic continuation.
- **Verify:** greedy generation reproduces the official reference continuation.
- **Optional afterward:** chat-template formatting and `fs chat`; useful UI, not
  part of the generation correctness gate.
- 📖 §2.2 (p.46) sampling · 🔧 `metal/{softmax,argsort}.metal`.

## M4 — KV cache  ☐  *(the "I made it faster" milestone)*
Cache K/V per layer; decode does one-token work. RAM-only first.
- **Artifact:** cached decode plus an uncached benchmark baseline.
- **Verify:** cached and uncached logits/tokens agree; report prefill and decode
  measurements with the baseline before claiming a speedup.
- 📖 §5.3 (p.136) · 🔧 `ds4_kvstore.c/.h`, `metal/dsv4_kv.metal` (SSD streaming = read-only study).

## M5 — Metal bring-up and end-to-end GPU execution  ☐
Bring up device/queue/buffers/pipelines, then move the complete inference path to
MSL through a deliberately bounded raw Objective-C/Metal FFI surface.
- **Artifact:** end-to-end generation executes on a modern Apple Silicon GPU.
- **Verify:** GPU checkpoints agree with the CPU oracle on the local Mac. Standard
  GitHub runners are not promised to execute Metal.
- 📖 §4.1 (p.96) incl. fusion (p.100), §3.1 (p.74), §3.5 (p.89) · 🔧 `ds4_metal.m` + all of `metal/`.
- **Note:** correctness and complete GPU execution come before optimization.

## M6 — Profile-driven Metal optimization and fusion  ☐
Profile the working GPU path, optimize measured bottlenecks, and fuse kernels only
where the data justifies it.
- **Artifact:** reproducible before/after local-Mac benchmarks.
- **Verify:** every optimized/fused path still agrees with CPU and unfused GPU
  paths; report hardware, prompt, dtype, prefill, and decode context.

## M7 — Quantization, conditional  ☐
Use the M4/M6 baselines to make an explicit benchmark-driven **go/no-go** decision.
If go, add a measured low-bit path and document quality, memory, and speed deltas;
if no-go, document the evidence and keep the clear full-precision path.
- **GGUF is separate:** optional interoperability work, not a quantization
  prerequisite and not part of the core completion gate.
- 📖 §5.1 (p.120) · 🔧 `gguf-tools/`, `gguf-tools/imatrix/`, dequant in `metal/*`.

## Post-core experiments  *(explicitly optional, not promised curriculum)*
Pick by interest only after M7:
- **Speculative decoding** (§5.2, p.129) — draft/target.
- **MoE** (§2.2.4, p.53) — routing + expert FFNs (`metal/moe.metal`).
- **DeepSeek-style compressed attention / MLA** (`dsv4_hc.metal`, `dsv4_kv.metal`;
  Raschka MLA notes) — the leap toward `ds4`'s actual model.
- **On-disk KV / SSD streaming** (`ds4_ssd.c`) — `ds4`'s signature idea.
- A tiny **server** or **REPL** for ergonomics.

---

## Parallel track — the docs (always-on)
After each milestone, write that milestone's doc — named for it, `mN-`:
`docs/m0-tokenizer.md` for M0, `docs/m1-weights.md` for M1,
`docs/m2-forward-pass.md` for M2: what we built, the math, the gotchas, and the
three-way cross-links (book §/page, `ds4` file, Raschka). Start in Markdown;
graduate the best ones to **rich HTML with diagrams** once content settles. Index
everything in [`docs/RESOURCES.md`](docs/RESOURCES.md).

**Learnings get their own home on the site.** The `docs/learnings/` notes are the
Markdown source of truth *and* graduate into a dedicated **Learnings** section on
the HTML site (its own nav entry + index), hand-distilled like the rest, linked
from the doc/milestone that references them (link the `.html`, not the raw `.md`).
HTML is where learnings earn nicer diagrams and *sparing* interactivity (à la
`diagrams.html`). See [`docs/dev-loop.md`](docs/dev-loop.md) → "Learnings → the
site's Learnings section" for the ritual. *(Notes `01–07`, `09`, `10` are graduated
to HTML; `08` is a stub awaiting M2.)*
