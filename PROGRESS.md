# PROGRESS — session log

> **Start every session here.** This is the running "where are we / what's next"
> log. Newest entry on top. The authoritative curriculum is [`PLAN.md`](PLAN.md);
> the big picture is [`docs/00-map.md`](docs/00-map.md).

**Current milestone:** M0 — Tokenizer (in progress: setup aligned; BPE not yet implemented)
**Engine status:** Rust scaffolding compiles (`cargo build` ✓); `src/tokenizer.rs` is an annotated sketch (`todo!()` bodies) with custom tokenizer errors. `fancy-regex` added for exact Qwen pre-tokenization; `serde_json` added for tokenizer JSON assets. Idempotent uv data pipeline live; golden vectors generated.
**Site:** live at <https://curtisalexander.github.io/fs/> (GitHub Pages from `/docs`).

---

## Session 4 — 2026-06-19 — Pre-coding alignment + dependency/testing decisions

**Did:**
- Aligned public setup docs before implementation: README status/layout/model choice,
  `PLAN.md` current milestone + doc numbering, `.gitignore` golden-fixture note.
- Added [`docs/dev-loop.md`](docs/dev-loop.md): start/end session ritual, local
  checks, uv oracle commands, site sync, and dependency policy.
- Added [`docs/testing.md`](docs/testing.md): verification philosophy, unit vs
  golden vs CLI vs benchmark checks, and M0 tokenizer testing plan.
- Added Rust dependency **`fancy-regex`** for Qwen's exact pre-tokenization regex.
- Added Rust dependency **`serde_json`** for tokenizer JSON assets; JSON parsing
  is not the tokenizer lesson.
- Added a hand-written `TokenizerError` enum + `Result<T>` alias and moved
  tokenizer public/helper signatures off `String` errors. `cargo test` ✓.
- Documented dependency freshness checks: Rust edition/toolchain check, Cargo
  update dry-runs, uv-only Python management, and uv's 7-day `--exclude-newer`
  age gate for Python updates.
- Added `tools/license-check.sh` and documented the license policy: MIT project,
  no GPL-family dependencies, manual review for weak-copyleft/unknown metadata.

**Decisions resolved this session:**
- **Dependency policy:** avoid dependencies that hide core inference concepts, but
  allow focused deps for non-core side problems when they improve correctness and
  avoid distracting side quests. ✅
- **M0 regex:** use `fancy-regex`; hand-writing BPE is core, hand-writing a
  Unicode/look-around regex engine is not. ✅
- **M0 JSON:** use `serde_json`; correctly parsing JSON is not the tokenizer
  lesson. ✅
- **M0 errors:** use a small custom `TokenizerError` enum right away, hand-written
  rather than `thiserror`, so failure modes stay visible and testable. ✅
- **M0 test shape:** use light inline unit tests for private helpers plus a light
  integration test over `tests/golden/tokenizer.json`. ✅
- **Milestone docs:** `docs/00-map.md` stays the map; milestone writeups start at
  `docs/01-tokenizer.md`, then `docs/02-weights.md`, etc. ✅
- **Python dependency management:** must use uv + `pyproject.toml`; no pip,
  requirements files, ad-hoc virtualenv setup, or inline script metadata. ✅
- **License policy:** reject GPL/AGPL/LGPL dependencies by default; warn/review
  weak-copyleft or unknown license metadata. ✅

**Still open for M0:** none before implementation.

**Next implementation order:**
1. `build_byte_encoder` — GPT-2 byte→unicode table; self-contained, no I/O.
2. `load_vocab` + `load_merges` — parse `vocab.json` / `merges.txt`.
3. `bpe` — the greedy merge loop (the heart of M0).
4. `pretokenize` — implement with Qwen's `fancy-regex` pattern.
5. Wire `encode` / `decode`; verify against `tests/golden/tokenizer.json` + round-trip.

---

## Session 3 — 2026-06-14 — M0 scaffolding + idempotent data pipeline

**Did:**
- **Rust scaffolding (edition 2024).** `Cargo.toml` = lib `fs` (the engine) +
  thin bin `fs` (the CLI), **zero dependencies**. `src/main.rs` hand-rolls argv
  parsing (no clap) and dispatches `tokenize` / `detokenize` / `help`.
  `src/lib.rs` → `pub mod tokenizer`. `cargo build` ✓.
- **Tokenizer annotated sketch** (`src/tokenizer.rs`): full struct (6 fields),
  3 public methods, 6 private helpers — real signatures + step-by-step
  pseudo-code in comments, all bodies `todo!()`. Reads as the *shape* of
  byte-level BPE before we implement. Documents the 4 encode stages
  (pre-tokenize → bytes→unicode → merge → look-up) and the `" hello world"`
  vs `"hello world"` leading-space behavior.
- **Idempotent + scriptable data pipeline (Python via uv).** `scripts/` is a
  uv project (`pyproject.toml` + `uv.lock`, managed CPython 3.13, deps
  `huggingface_hub` + `tokenizers`). `fetch_model.py` pulls Qwen3-0.6B tokenizer
  assets (~16 MB; `--weights` for the 1.5 GB later) — re-runs are no-ops.
  `gen_golden.py` runs the **official** tokenizer to emit `tests/golden/
  tokenizer.json` (14 tricky cases: leading spaces, CJK, emoji, code) — our M0
  correctness oracle (committed, so `cargo test` needs no Python).
- Extended `.gitignore` for Python (`.venv/`, `__pycache__/`). `models/` stays
  ignored; `scripts/uv.lock` + `tests/golden/` are committed.

**Decisions resolved this session:**
- **Python = a uv project** (`pyproject.toml` + lockfile), not PEP-723 inline
  scripts — one pinned, reproducible env shared across milestones. ✅
- **Fetch via `huggingface_hub`** (cached/resumable/idempotent), not raw curl. ✅
- **CLI = hand-rolled, zero-dep** arg parsing (ds4 "no hidden abstraction" ethos). ✅
- Python is **only ever a one-shot oracle** (golden data), never a 2nd engine. ✅

**Open decisions for M0 (pick up before/while implementing):**
1. **Pre-tokenization regex.** Qwen's pattern needs Unicode classes (`\p{L}`,
   `\p{N}`) **and** a negative look-ahead (`\s+(?!\S)`). The `regex` crate does
   classes but not look-ahead; `fancy-regex` does both. Choose: **(a)** add
   `fancy-regex` (exact, one dep) vs **(b)** hand-roll the splitter (zero deps,
   more code + Unicode tables). This is where "zero deps" gets tested.
2. **`Tokenizer::load` error type.** Currently `Result<Self, String>` (cheap,
   keeps CLI compiling). Optionally graduate to a real error enum
   (missing-file / bad-JSON / malformed-merge) — small idiomatic-Rust lesson.

**M0 implementation order (bottom-up, each unit-tested vs the golden file):**
1. `build_byte_encoder` — GPT-2 byte→unicode table; self-contained, no I/O. *(start here — independent of the regex decision)*
2. `load_vocab` + `load_merges` — parse `vocab.json` / `merges.txt`.
3. `bpe` — the greedy merge loop (the heart of M0).
4. `pretokenize` — **after** resolving open decision #1.
5. Wire `encode` / `decode`; verify against `tests/golden/tokenizer.json` + round-trip.

**Method note:** continue sketch-first → read → implement-together, one helper at
a time; pull the best explanations into `docs/01-tokenizer.md` + `learnings/`.

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
