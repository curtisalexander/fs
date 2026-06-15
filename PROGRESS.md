# PROGRESS — session log

> **Start every session here.** This is the running "where are we / what's next"
> log. Newest entry on top. The authoritative curriculum is [`PLAN.md`](PLAN.md);
> the big picture is [`docs/00-map.md`](docs/00-map.md).

**Current milestone:** M0 — Tokenizer (not started)
**Engine status:** scaffolding only, no code yet.
**Site:** live at <https://curtisalexander.github.io/fs/> (GitHub Pages from `/docs`).

---

## Session 2 — 2026-06-13 — HTML learning site + logo + doc alignment

**Did:**
- **Published a GitHub Pages site** at <https://curtisalexander.github.io/fs/> —
  served straight from `main` `/docs` (Settings → Pages → deploy-from-branch), with
  `docs/.nojekyll` so our hand-written HTML is served verbatim. **No CI/Action.**
- **HTML spine** `docs/index.html`: a distillation of `00-map.md` — interactive
  abstraction ladder (click-to-expand rungs + "highlight what we build" toggle),
  data-journey pipeline, prefill/decode split, correct→fast→small cards, coverage
  table. Custom design system in `docs/assets/css/main.css` (themeable, warm
  "brown-dwarf" dark palette, no framework); vanilla JS in `docs/assets/js/app.js`.
- **Sync model = "distillation, not conversion":** HTML is hand-authored, not
  auto-generated. `tools/sync-ledger.tsv` records which markdown each page distills
  + the reconciled commit; `tools/sync-check.sh` reports drift (and `--update`
  re-stamps). Each page also self-declares sources via `<meta name="fs-distills">`.
- **Logo:** "failed star" = a sad, hunched chibi star in a faint brown-dwarf glow
  with orbiting motes + a fizzled spark. Brand/hero = `docs/assets/logo/star.svg`;
  favicon = mono `star-mono.svg`. Candidates + contact sheet kept in
  `assets/logo-drafts/` for future iteration.
- **Doc alignment** (README/RESOURCES/prerequisites): reframed "three sources" as
  the *spine* (vs the wider brush-up set); added Kiely peruse link
  (inferenceengineering.tech → Baseten Books); split Raschka's *free articles*
  (load-bearing) from his *book* (optional, paid); added an inference-specific
  reading list (Weng, kipply, EleutherAI, Horace He, Grootendorst, Bekman).

**Decisions resolved this session:**
- Publish via **branch-folder Pages, not GitHub Actions** (less machinery, more
  transparent — matches the "no hidden abstraction" ethos). Site lives in `/docs`
  alongside the working markdown; `.nojekyll` keeps the `.md` inert.
- HTML ≠ markdown auto-conversion; keep them "kinda in sync" via **drift detection**.

**Continued (same day) — diagrams, polish, logo:**
- **Four interactive diagrams** now live on `diagrams.html`, each real math on toy
  data: **tokenizer** (toy BPE, M0), **sampling** (softmax + temp/top-k/top-p, M3),
  **attention** (scaled-dot-product + causal-mask toggle, M2), **KV cache** (decode
  stepper, linear-vs-quadratic tally, M4). Logic in `assets/js/diagrams.js`.
- **`prerequisites.html`** distilled; **light/dark toggle** added site-wide
  (tokenized palette, no-flash init respecting `prefers-color-scheme`, persisted).
- **Logo finalized:** chibi sad-star + brown-dwarf glow/motes/spark
  (`docs/assets/logo/star.svg`); mono favicon; added to README header.
- **Tone + clarity pass:** cut over-the-top lines; citation chips now name the book
  ("Inference Engineering (Kiely)"); renamed the 🔵 tier "We'll teach you" →
  "What we'll demonstrate"; reworked the hero tagline.
- Pref noted in memory: **always use `rg`**, not grep.

**Next session (M0 — tokenizer):** *(unchanged from below)*

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
- Wrote `docs/prerequisites.md` — the "what to know before diving in" gate (tiered
  floor/helpful/we'll-teach, brush-up resources, map→knowledge table, a note on the
  unusual Rust we'll explain inline).
- Pushed scaffold to remote `origin` (github.com/curtisalexander/fs), preserving
  the existing MIT LICENSE.

**Next session (M0 — tokenizer):**
1. Grab Qwen3-0.6B's tokenizer files from HF (`tokenizer.json`, config) into
   `models/` (git-ignored).
2. `cargo init` the Rust project; lay out `src/`.
3. Implement BPE encode/decode in Rust against Qwen's real vocab/merges.
4. Verify token IDs match the official tokenizer on a string set; write
   `docs/01-tokenizer.md` (cross-link book §2.2, ds4 `ds4.c` BPE + hash table,
   Raschka BPE; see also `learnings/02-radix-tree.md`).
