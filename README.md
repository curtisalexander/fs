<p align="center">
  <img src="docs/assets/logo/star.svg" width="150"
       alt="Failed Star logo: a sad, hunched cartoon star in a faint brown-dwarf glow">
</p>

<h1 align="center">Failed Star (<code>fs</code>)</h1>

<p align="center"><em>A small, self-contained LLM inference engine for Apple Silicon —<br>
built from scratch, in the open, to <strong>learn and teach</strong> how inference engineering works.</em></p>

<p align="center">
  <a href="https://curtisalexander.github.io/fs/"><strong>📖&nbsp;Read the learning site →</strong></a>
</p>

A **failed star** (a brown dwarf) is smaller than a dwarf star: not enough mass
to sustain fusion. This project is the smaller sibling of
[**Dwarf Star (`ds4`)**](https://github.com/antirez/ds4), antirez's
self-contained inference engine for DeepSeek-V4. Where `ds4` targets big MoE
models on 96GB+ Macs, **Failed Star runs a *tiny* model on a 64GB MacBook Pro
(M5)** — and trades raw capability for something else: every line is meant to be
*read, understood, and learned from*.

## Why this exists

The goal is **understanding inference, by building it.** Reading about attention
is one thing; writing the kernel that computes it and watching tokens stream out
of your own code is another. This repo is the second thing.

Three sources form its **spine**, cross-referenced throughout the docs. (The
[prerequisites](docs/prerequisites.md) point to a wider set of optional brush-up
and go-deeper resources — those fill gaps; these three are what the docs lean on.)

1. **The concepts** — *Inference Engineering* by Philip Kiely (Baseten, 2026).
   The "why" and the vocabulary. (Peruse the free
   [interactive guide](https://inferenceengineering.tech/), or get your own copy
   from [Baseten Books](https://www.baseten.co/inference-engineering/);
   `Inference Engineering.pdf` is in this repo.)
2. **A real implementation** — [`ds4`](https://github.com/antirez/ds4), cloned
   into `reference/ds4/`. The "how a pro does it." *Working code doesn't lie.*
3. **Architecture context** — Sebastian Raschka's *free articles*: the
   [architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison),
   [gallery](https://sebastianraschka.com/llm-architecture-gallery/), and
   [workflow for understanding LLMs](https://magazine.sebastianraschka.com/p/workflow-for-understanding-llms).
   (His *book* is a good optional extra, not a dependency — see the prerequisites.)

## How it's built

- **Host language: Rust.** Model loading, tokenizer, orchestration, sampling, KV
  cache — all Rust.
- **GPU kernels: MSL** (Metal Shading Language) — hand-written, one operation per
  file, just like `ds4`'s `metal/` shaders.
- **Metal via raw FFI / the Objective-C runtime — no convenience wrapper crate.**
  We send messages to Metal ourselves so nothing is hidden. Tight, like `ds4`.
- **First model: Qwen3-0.6B** — a tiny dense model with GQA, RoPE, SwiGLU, and
  RMSNorm; small enough to inspect and debug while still looking like a real
  modern LLM.
- **Correctness via golden vectors:** match logits from the model's official
  implementation. (Python appears *only* as a one-shot oracle, never as a second
  engine.)

## Repo layout

```
fs/
├── README.md                  ← you are here
├── PLAN.md                    ← the milestone curriculum (M0 … M7+)
├── PROGRESS.md                ← running session log; start here each session
├── Inference Engineering.pdf  ← local copy of the book (ignored; bring your own)
├── src/                       ← Rust engine + thin CLI
├── scripts/                   ← uv-managed Python oracle/data scripts
├── tests/golden/              ← committed golden fixtures for verification
├── tools/                     ← site/sync helper scripts
├── docs/                       ← the learning site + notes (served at /fs via Pages)
│   ├── index.html             ← learning-site landing page (rich HTML)
│   ├── prerequisites.md       ← what to know before diving in (read this first)
│   ├── 00-map.md              ← THE BIG PICTURE of an inference engine
│   ├── 01-tokenizer.md        ← M0 writeup (.md + rich .html version)
│   ├── dev-loop.md            ← how to resume work after a break
│   ├── testing.md             ← verification strategy and golden-vector plan
│   ├── diagrams.html          ← shared diagram gallery
│   ├── RESOURCES.md           ← cross-reference index (book §§, ds4 files, Raschka)
│   ├── learnings/             ← bite-sized notes on what we figured out & why
│   └── assets/                ← logo + site assets
├── reference/ds4/             ← antirez's ds4 — pinned git submodule (read-only ref)
└── models/                    ← downloaded model assets (ignored; generated locally)
```

## Where to start

1. Read **[`docs/prerequisites.md`](docs/prerequisites.md)** — the honest "what to
   know before you dive in" (spoiler: inference is the forward pass only — no
   training, no calculus), with brush-up resources and a knowledge-map.
2. Read **[`docs/00-map.md`](docs/00-map.md)** — the end-to-end picture of an
   inference engine, with an "abstraction ladder" so you can stop digging at
   whatever depth interests you.
3. Skim **[`PLAN.md`](PLAN.md)** — the milestones.
4. Each session, open **[`PROGRESS.md`](PROGRESS.md)** to see what's next.
5. If resuming development, use **[`docs/dev-loop.md`](docs/dev-loop.md)** and
   **[`docs/testing.md`](docs/testing.md)** for the local checks and verification
   strategy.

## Status

🌱 **M0 — Tokenizer: ✅ done. M1 — Load the weights: ✅ done. M2 — Forward pass:
next.** `fs inspect models/qwen3-0.6b` loads `config.json` + `model.safetensors`,
derives the expected tensor set from the config, cross-checks the file against it,
and prints a shape-first legend + tensor table + verdict — the real model checks
clean (311 tensors, 596M logical params; see [`docs/02-weights.md`](docs/02-weights.md)).
The weights are mmap'd zero-copy via raw POSIX FFI, bf16 kept lazy. Next step: the
forward pass (embeddings → transformer blocks → logits), CPU-first.

**Milestones** (the full curriculum, with cross-links, lives in [`PLAN.md`](PLAN.md)):

- [x] **M0 — Tokenizer** — text ↔ token IDs, verified against the real vocab
- [x] **M1 — Load the weights** — `fs inspect`; tensor set cross-checked vs the config
- [ ] **M2 — Forward pass → logits** ← *current*
- [ ] M3 — Sampling → generation
- [ ] M4 — KV cache
- [ ] M5 — Quantization
- [ ] M6 — Metal acceleration
- [ ] M7+ — Stretch goals

This is a slow, multi-session learning project. It is not (yet) fast, capable, or
finished — that's the point. Local models keep getting better; the bet is that a
clean, well-documented small engine becomes *more* useful to *more* people over
time.
