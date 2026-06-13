# PROGRESS — session log

> **Start every session here.** This is the running "where are we / what's next"
> log. Newest entry on top. The authoritative curriculum is [`PLAN.md`](PLAN.md);
> the big picture is [`docs/00-map.md`](docs/00-map.md).

**Current milestone:** M0 — Tokenizer (not started)
**Engine status:** scaffolding only, no code yet.

---

## Session 1 — 2026-06-13 — Scaffolding & big-picture map

**Did:**
- Researched all three sources: read the book's TOC/structure (it's a
  *production-serving* survey — concepts + vocabulary, not build-from-scratch),
  cloned and inspected `ds4` (Metal is its primary backend: `ds4_metal.m` ≈ 26k
  lines, 19 MSL shaders in `metal/`; runs DeepSeek-V4-Flash 284B/13B MoE), and
  read Raschka's workflow + architecture comparison.
- Locked the approach (see PLAN "Decisions locked"): **Rust host + MSL kernels,
  raw Metal FFI (no wrapper crate), tiny dense model, golden-vector validation,
  Metal-only scope.**
- Named the project **Failed Star (`fs`)** — a brown dwarf, smaller than `ds4`.
- Wrote: `README.md`, `docs/00-map.md` (the big-picture abstraction ladder +
  data-journey + cross-reference map), `PLAN.md` (M0–M7+), `docs/RESOURCES.md`,
  this log.

**Decisions resolved this session:**
- ds4 = **pinned git submodule** @ `d881f2a` (MIT licensed). ✅
- Starter model = **Qwen3-0.6B** (tiny dense: GQA + RoPE + SwiGLU + RMSNorm). ✅
- On-disk format = **safetensors for M1–M4, GGUF at M5.** Qwen ships safetensors
  natively; GGUF (what ds4 uses) arrives with the quantization lesson. ✅
- Repo `git init`'d (branch `main`).
- Wrote first learning note: `docs/learnings/01-safetensors-vs-gguf.md`.
- Pushed scaffold to remote `origin` (github.com/curtisalexander/fs), preserving
  the existing MIT LICENSE.

**Next session (M0 — tokenizer):**
1. Grab Qwen3-0.6B's tokenizer files from HF (`tokenizer.json`, config) into
   `models/` (git-ignored).
2. `cargo init` the Rust project; lay out `src/`.
3. Implement BPE encode/decode in Rust against Qwen's real vocab/merges.
4. Verify token IDs match the official tokenizer on a string set; write
   `docs/01-tokenizer.md` (cross-link book §2.2, ds4 `rax.c`, Raschka BPE).
