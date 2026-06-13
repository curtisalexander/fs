# Prerequisites — what to know before you dive into the code

**Read this before [`00-map.md`](00-map.md) and before M0.** This is a *learning*
repo, so the bar is deliberately low — but not zero. Here's the floor, what's
merely helpful, and what we plan to demonstrate. Then a "brush-up" resource list
and a knowledge-map you can use to find your own gaps.

> ### The single most liberating fact
> **Inference is the forward pass only.** We are *running* a model that someone
> else already trained. That means **no training, no backpropagation, no
> gradients, no calculus.** If you've avoided ML because of the math of *learning*,
> almost none of it applies here. Inference is: look up some vectors, multiply
> matrices, normalize, pick the most likely next token, repeat. That's the whole
> game.

---

## How to read the tiers

- 🟢 **The floor** — have a *feel* for these before starting. If one is shaky,
  spend an hour with a resource below; you don't need mastery.
- 🟡 **Helpful — brush up as you go.** Nice to have seen once; you can learn it
  *while* following along.
- 🔵 **What we'll demonstrate.** Come curious, be ready to dig in. Showing up knowing these
  would defeat the purpose.

---

## 🟢 The floor (have some feel for these)

| Thing | What "enough" looks like | Brush up with |
|---|---|---|
| **Vectors & matrix multiplication** | You know a matmul is rows-dot-columns, and why shapes have to line up (`[m×k]·[k×n] = [m×n]`). This is ~80% of what an LLM *does*. | 3Blue1Brown *Essence of Linear Algebra* (esp. matrix multiplication) |
| **What a neural net / forward pass is** | Inputs → weighted sums → a nonlinearity → outputs, stacked in layers. You don't need to know how it's *trained*. | 3Blue1Brown *Neural Networks* ch. 1–2 |
| **Basic Rust** | Read & write simple programs: `struct`/`enum`, `Vec`, slices, `Option`/`Result`, ownership & borrowing basics, `match`. Not: async, macros, lifetimes-gymnastics. | The Rust Book (ch. 1–10), `rustlings` |
| **Command line + git** | clone, branch, commit; run a binary; navigate folders. | (you're already here) |
| **Bytes & number types** | What `f32`/`f16`/`bf16`/`int8` are, roughly; that an array is just numbers laid out in memory. | our [learning 01](learnings/01-safetensors-vs-gguf.md) |

## 🟡 Helpful — brush up as you go

| Thing | Why it helps | Brush up with |
|---|---|---|
| **The transformer, conceptually** | The shape of the thing you're building: tokens → embeddings → attention + feed-forward (×N) → logits. | Jay Alammar, *The Illustrated Transformer*; 3B1B *attention* videos |
| **Softmax & nonlinearities** | Softmax turns scores into probabilities (used in attention *and* sampling); SiLU/GELU are the FFN's nonlinearity. | book §2.1.2 (p.44) |
| **Probability / sampling basics** | "A distribution over the next token," temperature, top-k/top-p. | Karpathy, *Let's build GPT* (sampling part) |
| **Memory layout / mmap / FFI (idea only)** | Why row-major contiguous arrays matter; what "map a file into memory" and "call C/Obj-C from Rust" mean. | learning 01; we go deeper at M1/M6 |

## 🔵 What we'll demonstrate (don't pre-study these)

Attention mechanics in detail · **RoPE** (positions) · **RMSNorm** · **GQA**
(grouped-query attention) · **SwiGLU** FFN · the **KV cache** · **quantization** ·
**Metal/GPU kernels, MSL, and the FFI plumbing** · **BPE tokenization** internals.
Each is a milestone with its own doc. Curiosity > preparation.

### A note on the Rust we use

We assume *basic* Rust (the 🟢 floor). But a from-scratch inference engine reaches
into corners of the language a typical app never touches — and we **explain those
inline, and in learning notes, the first time they appear.** Never pre-learn this
list; just know it's coming and it'll be explained:

- **`unsafe`, raw pointers (`*const T` / `*mut T`), and `extern "C"`** — to call
  Metal/Objective-C and to read memory-mapped bytes.
- **`#[repr(C)]`, layout & alignment** — to match on-disk file formats and GPU
  buffer layouts exactly.
- **Bit-level number twiddling** — e.g. decoding **bf16** (and later quant blocks)
  into `f32` by hand.
- **`mmap` + slices over foreign memory** — zero-copy weight loading.
- *(later, maybe)* **SIMD intrinsics** for CPU kernels.

So if you hit Rust that looks nothing like the Rust Book, that's expected — it
comes with an explanation. The *odd* Rust is part of what this repo teaches.

---

## Brush-up resources (ranked by usefulness for *this* repo)

**See it (intuition, ~a few hours total):**
- **3Blue1Brown — Neural Networks series** (incl. the GPT/attention videos):
  <https://www.3blue1brown.com/topics/neural-networks> — the best visual intuition
  for matmul, neural nets, and attention. Start here if the math feels far away.
- **Jay Alammar — The Illustrated Transformer:**
  <https://jalammar.github.io/illustrated-transformer/> — the classic picture of
  the architecture we're implementing.

**Code it (the closest match to our method — "working code doesn't lie"):**
- **Andrej Karpathy — Neural Networks: Zero to Hero:**
  <https://karpathy.ai/zero-to-hero.html>. Two videos are directly on-point:
  *"Let's build the GPT Tokenizer"* (primes our **M0**) and *"Let's build GPT from
  scratch"* (primes **M2/M3**).
- **Karpathy — `llama2.c`:** <https://github.com/karpathy/llama2.c> — a *single
  ~970-line C file* that runs Llama inference end-to-end. A **much gentler full
  engine to read than `ds4`** — think of it as a stepping stone toward `ds4`, and a
  close cousin of what *we're* building (just in C, not Rust/Metal).
- *(optional, paid)* **Sebastian Raschka — Build a Large Language Model (From
  Scratch)** (book; free code at <https://github.com/rasbt/LLMs-from-scratch>) — a
  clean book-length walkthrough of tokenization, attention, and the transformer. It
  covers the same "build it from scratch" ground as Karpathy's free videos above —
  reach for it only if you want the long-form, sit-down treatment.

**Place it in context (architecture):**
- Raschka, [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
  and the [gallery](https://sebastianraschka.com/llm-architecture-gallery/) — read
  *lightly* now (to see where Qwen3 and DeepSeek-V4 sit); revisit per milestone.

**Go deeper on *inference* specifically (free, by noted practitioners):**
Inference reuses a model someone else trained, but it has its own arithmetic —
memory- vs compute-bound, KV-cache cost, where the time actually goes. These are
the canonical free reads, and most are ongoing series worth following:
- **Philip Kiely — *Inference Engineering* interactive guide:**
  <https://inferenceengineering.tech/> — the free companion to our concept book;
  animated diagrams + calculators (VRAM, arithmetic intensity, KV-cache sizing).
- **Lilian Weng (Lil'Log) — "Large Transformer Model Inference Optimization":**
  <https://lilianweng.github.io/posts/2023-01-10-inference-optimization/> — the best
  single free survey of inference optimization; her whole blog is a rigorous series.
- **kipply (Carol Chen) — "Transformer Inference Arithmetic":**
  <https://kipp.ly/transformer-inference-arithmetic/> — the canonical back-of-the-
  envelope: why decode is memory-bound and how big the KV cache gets. Primes M4/M6.
- **EleutherAI — "Transformer Math 101":** <https://blog.eleuther.ai/transformer-math/>
  (+ the [cookbook](https://github.com/EleutherAI/cookbook)) — companion FLOP/memory math.
- **Horace He — "Making Deep Learning Go Brrrr From First Principles":**
  <https://horace.io/brrr_intro.html> — compute- / memory- / overhead-bound, the
  mental model behind every kernel choice. Primes M6.
- **Maarten Grootendorst — "A Visual Guide to Quantization"** (+ his MoE guide):
  <https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization> —
  50+ visuals on GPTQ/GGUF/BitNet. Primes M5 (and M7 for MoE).

**Another practitioner book (free, optional):**
- **Stas Bekman — "Machine Learning Engineering Open Book":**
  <https://github.com/stas00/ml-engineering> — a living, open practitioner book;
  strong on the serving / scaling / inference-ops side beyond a single model.

**The book itself (our concept source) — the prerequisite chapters:**
- **Ch 0 "Inference" (p.15)** — what inference even is.
- **Ch 1 "Prerequisites" (p.23)** — yes, literally; scale, latency/throughput.
- **Ch 2.1 "Neural Networks" (p.42)** and **§2.2 "LLM Inference Mechanics" (p.46)**
  — the vocabulary every later milestone assumes.
- *(Optional, for the "why")* **§2.4 "Bottlenecks" (p.61)** — why optimizations exist.

**Rust (if it's rusty):**
- The Rust Book: <https://doc.rust-lang.org/book/> · `rustlings`:
  <https://github.com/rust-lang/rustlings>

---

## The map → knowledge (which rung needs which idea)

This expands the abstraction ladder in [`00-map.md`](00-map.md): for each rung, the
concepts to have *some* feel for, and the tier. Use it to find your gaps — if a rung
interests you, make sure its "concepts" aren't a total mystery.

| Rung (from the map) | Concepts to have a feel for | Tier |
|---|---|---|
| **7 Chat / agent loop** | prompts, chat templates, "a conversation is just tokens" | 🟡 |
| **6 Generation loop** | probability distribution over tokens, sampling, temperature | 🟡 |
| **5 Model forward pass** | layers, embeddings, logits, "it's mostly matmuls" | 🟢🟡 |
| **4 Transformer block** | attention (Q/K/V), feed-forward, residual add, normalization | 🟡🔵 |
| **3 Tensor operations** | matmul, softmax, vectors, **shapes/dimensions** | 🟢 |
| **2 Kernels** | "a function run over an array, in parallel"; light GPU idea | 🔵 |
| **1 Memory & numbers** | dtypes, bytes, contiguous arrays, mmap | 🟢🟡 |
| **0 Hardware** | CPU vs GPU, threads, memory is slower than compute | 🔵 |

Notice the pattern: the **floor (🟢) lives at the bottom and middle** (numbers,
shapes, matmul, forward pass) — the concrete stuff. The **fancy named things
(🔵)** — attention, RoPE, kernels — are what we *build and explain*. You can climb
in from the middle.

---

## A suggested ramp before M0 (skip what you already have)

1. **(~1 hr, optional)** Watch 3B1B's attention video *or* read the Illustrated
   Transformer — just to hold the whole shape in your head.
2. **Read** book **Ch 0** and **§§2.1–2.2** — the concepts + vocabulary.
3. **Skim** `reference/ds4/README.md` and `MODEL_CARD.md` — see the destination
   (don't try to read the 26k-line `ds4.c`!).
4. **(if Rust is rusty)** a few `rustlings` exercises or Rust Book ch. 1–10.
5. **(optional, primes M0)** Karpathy's *"Let's build the GPT Tokenizer."*

Then open [`../PLAN.md`](../PLAN.md) and start **M0 — the tokenizer**.

---

## Self-check (you're ready when…)

You don't need to *answer* these — just feel that they're not total fog:

- "An LLM mostly multiplies matrices" doesn't sound mysterious.
- You could write a Rust function that takes a `&[f32]`, sums it, and returns the result.
- You know a model file is "numbers + a table describing them" (see learning 01).
- You accept that we'll explain attention/RoPE/KV-cache/Metal when we get there.

If those feel roughly OK, **you're ready.** If not, the floor table above tells you
exactly which hour to spend first.
