# PROGRESS — session log

> **Start every session here.** This is the running "where are we / what's next"
> log. Newest entry on top. The authoritative curriculum is [`PLAN.md`](PLAN.md);
> the big picture is [`docs/00-map.md`](docs/00-map.md).

**Current milestone:** M1 — Load the weights (◐ building, bottom-up). **First helper landed:** `Mmap::open` — the raw POSIX `mmap` FFI — implemented + verified. Design (locked earlier): mmap via raw FFI, lazy bf16 (raw bytes, convert on access), shape-first `fs inspect` cross-checking the tensor set against `config.json`. **Next helper: `SafeTensors::load`** (u64 header len → JSON header → tensor directory). M0 — Tokenizer ✅ complete (14/14 golden).
**Engine status:** `fs tokenize` / `fs detokenize` run end-to-end against Qwen3-0.6B (tokenizer in `src/tokenizer.rs`, all four BPE stages + special tokens). **M1 in progress:** `src/safetensors.rs::Mmap::open` maps the whole file zero-copy via raw `mmap`/`munmap` FFI with RAII `munmap` on `Drop`; 3 unit tests exercise the live syscall (byte-exact round-trip incl. NUL/high bytes, missing-file → typed `MapFailed`, empty-file rejected). **20 unit + 2 golden integration tests green; clippy clean.** Still `todo!()` in M1 order: `SafeTensors::load` → `Config::load` → `expected_tensors`/`cross_check` → `render_*`. Milestone writeups: [`docs/01-tokenizer.md`](docs/01-tokenizer.md); `docs/02-weights.md` owed at M1 close.
**Site:** live at <https://curtisalexander.github.io/fs/> (GitHub Pages from `/docs`). **Learnings now have their own section** (`docs/learnings/index.html` + a page per note, in the site nav) — the owed HTML-graduation debt for `learnings/01–07` is **cleared**.

---

## Session 10 — 2026-07-06 — M1 first helper: `Mmap::open` (raw POSIX mmap FFI)

**Did:** implemented the bottom of the M1 stack — the one primitive every later
helper reads through — with a strong teach-through, then verified it against the
real syscall. Kept scope to exactly this one helper.

**`Mmap::open` (`src/safetensors.rs`):** path → zero-copy, read-only mapping of the
whole file.
- open file → fd; map length = `metadata().len()`; guard `len == 0` (mmap rejects
  it with `EINVAL`) *before* the syscall so the error reads "empty file".
- the single `unsafe` line is the `mmap()` call itself — unsafety is *contained*,
  not spread; safe Rust above it, the raw pointer never escapes `Mmap`.
- two POSIX gotchas made loud: failure is **`MAP_FAILED` = `(void*)-1`, not null**
  (checked, with `last_os_error()` for the errno); and the **fd can drop after
  mapping** — a live mapping holds its own reference to the file, so `file` is let
  go at function end without tearing the mapping down.
- derived `Debug` on `Mmap` so tests can `unwrap_err()` the `Ok`-side.

**Verify (M0 cadence — one helper, tests that hit reality):** 3 unit tests exercise
the live FFI — byte-exact round-trip (feeds `\x00\x01\xfe\xff` to prove *raw bytes*,
not a decoded string) · missing file → typed `MapFailed` (no panic) · empty file
rejected. Full suite **20 unit + 2 golden green**; `cargo clippy --all-targets`
clean. Committed `814905d`, pushed to `origin/main`.

**Teaching notes already in place** (Session 9): [`learnings/06-mmap.md`](docs/learnings/06-mmap.md)
(virtual memory / lazy paging / RAII) and its HTML page back the "why."

**Next:** `SafeTensors::load` — the first *reader* on the mapping: first 8 bytes =
`u64` LE header length `N` → parse `[8, 8+N)` as the JSON header (`serde_json::Value`)
→ build the tensor directory (name→`{dtype, shape, [start,end)}`), validating each
entry's byte length against `shape · dtype.size()`. Then `Config::load`.

---

## Session 9 — 2026-07-06 — M1 teaching prep: bf16 note + graduate all Learnings to HTML

**Did:** closed the pre-coding teaching gaps before touching M1 engine code, then
paid off the whole Learnings-→-site debt in one batch.

**New learning notes (Markdown):**
- `docs/learnings/07-bf16.md` — the weights' number format. bf16 = fp32's top 16
  bits (same 8-bit exponent, 7-bit mantissa), so widening is a free/exact 16-bit
  shift; quantifies the "lazy bf16" decision (eager convert = 1.4 GB → ~2.8 GB) and
  separates bf16 (shift) from M5 quantization (decode). Answers the "why defer
  conversion?" question `fs inspect`'s `dtype BF16` output will raise.
- `docs/learnings/08-row-major-strides.md` — **stub** (🌱), deliberately deferred to
  M2: shape→byte-offset, row-major layout, strides, indexing the blob. Placeholder
  so the teaching moment isn't missed when M2 first reads a weight element-by-element
  (embedding gather → matmul). Both indexed in `learnings/README.md`.

**Learnings graduated to HTML (the owed debt, done):**
- Decision (asked + locked): **hand-distill each note to bespoke HTML** (policy =
  "distillation, not conversion"), upgrading ASCII diagrams to real theme-aware SVG.
- Built the **Learnings site section**: `docs/learnings/index.html` (card grid) +
  a "Learnings" entry in the nav on all pages (index/prerequisites/diagrams/
  01-tokenizer + every note). Added a small Learnings CSS block to `main.css`
  (`.note-meta`, `.figure`/`.legend`, `.crosslinks`, `.learn-card`) — no new
  framework, tokens reused.
- Distilled **01–07** to HTML (`07-bf16.html` hand-authored as the canonical
  template with an SVG bit-field diagram; 01–06 fanned out to parallel subagents
  against that template + strict class whitelist, then QA'd). SVGs rendered:
  01 safetensors layout · 02 radix tree · 04 embedding forward-pass · 05 **four**
  (`[out,in]` conv, residual bus, head split, GQA) · 06 read-vs-mmap flow · 07
  bf16 fields. 03 used tables/splits (better fit than SVG).
- Wired links: milestone page now links `learnings/03-bpe.html` (was `.md`); each
  note's cross-links point at sibling `.html`; registered all pages in
  `tools/sync-ledger.tsv`.
- **QA:** tag balance (section/div/svg) all matched across 8 pages; no page-level
  `<style>` (only SVG-internal); every internal `.html` link resolves site-wide;
  `sync-check.sh` reports **in sync**.

**Decisions resolved this session:**
- **bf16 documented now, not at M2** — it grounds M1's "lazy bf16" decision. ✅
- **row-major/strides deferred to M2** as a stub — it's an *indexing* lesson that
  belongs with the matmul, not the load. ✅
- **Learnings = hand-distilled HTML** (not md→HTML auto-gen), matching the hero
  pages' "distillation not conversion" model. ✅

**Note:** ledger rows stamped at HEAD `99b6640`; on the commit that lands this
batch, run `tools/sync-check.sh --update` to re-stamp to the new commit.

**Next (unchanged):** M1 engine code, bottom-up from `Mmap::open` (the raw POSIX
FFI) → `SafeTensors::load` → `Config::load` → `expected_tensors`/`cross_check` →
`render_*`, verified against the real `model.safetensors`. Then `docs/02-weights.md`
(+ graduate `learnings/08` when M2 writes it).

---

## Session 8 — 2026-06-24 — M1 kickoff: design dialogue, AGENTS.md, learnings-site policy

**Did:** opened M1 (load the weights) with a scaffolding-first design dialogue
(no engine code yet) and locked the milestone's shape.

**M1 design decisions locked:**
- **Storage = mmap via raw FFI**, implemented *now* (not deferred). It's the
  zero-copy "the file *becomes* memory" lesson, fits the raw-FFI ethos + `ds4`,
  and becomes its own teaching moment.
- **bf16 = lazy.** Tensors stay raw bf16 bytes in the mapping; checksums run over
  raw bytes (proves we read the right region). A `bf16_to_f32` helper gets written
  but is only *called* in M2 — eager conversion would copy 1.4 GB → ~3 GB of f32
  and defeat mmap.
- **`fs inspect` = shape-first + config cross-check.** Output leads with a
  dimension legend (V/H/L/d/heads/I), a grouped-by-layer table with an
  `in ──▶ out` column so the residual stream is readable, and a cross-check that
  asserts every Linear's `in` lines up with the architecture (q/k/v ← H, o ← 16·d,
  gate/up ← H, down ← I; embed = [V,H]; tied lm_head) + total params (~596M,
  embeddings ≈ 26%).
- **Shape-clarity is a standing priority** (saved to agent memory): whenever shapes
  appear — docs, CLI, code — make them explicit and *visible*, backed by asserts.
  Learning-first; "make it fast" is a separate later lesson.

**Scaffold landed (sketch-first, `todo!()` bodies, `cargo build` + `clippy` green):**
- `src/config.rs` — `Config` (the 7 named dims + derived `q_width`/`kv_width`/
  `gqa_group`) + `ConfigError`. Parses `config.json` via `serde_json::Value` by
  hand (no `serde` derive, no new dep); graduating to a derive is an open option.
- `src/safetensors.rs` — `Dtype`/`Tensor`/`SafeTensors` + `SafeTensorsError`, and
  the **raw mmap FFI** (`unsafe extern "C"` `mmap`/`munmap`, no `libc` crate) behind
  a small RAII `Mmap` (unmaps on `Drop`). `bf16_to_f32` written but unused until M2.
- `src/inspect.rs` — `expected_tensors` (the config-derived spec), `cross_check`,
  `render_legend`/`render_table`/`render_verdict`, `run`, `InspectError`.
- `main.rs`/`lib.rs` — `fs inspect [DIR]` wired (panics via `todo!` until filled);
  modules exported.
- Docs: `docs/learnings/05-reading-shapes.md` (the shapes lesson) and
  `docs/learnings/06-mmap.md` (mmap as its own note, per decision below); both
  indexed in `docs/learnings/README.md`. Markdown-first; HTML graduation later.

**Decision:** **mmap FFI is its own learning note** (`06-mmap.md`), not folded into
the milestone writeup — it's a standalone teaching topic (virtual memory, lazy
paging, raw POSIX FFI, RAII cleanup).

**Process / docs decisions:**
- **AGENTS.md + symlink.** Added a tight [`AGENTS.md`](AGENTS.md) operating
  contract and symlinked `CLAUDE.md → AGENTS.md` so Claude Code auto-loads it every
  session (it does *not* read AGENTS.md natively; it follows the symlink). AGENTS.md
  points at `PROGRESS.md`/`PLAN.md`/`dev-loop.md` and lists the core invariants — it
  does not inline them.
- **Learnings get their own site section.** Policy: learnings stay Markdown-source
  but graduate into a dedicated **Learnings** HTML section (own nav + index),
  linked from the referencing doc (`.html`, not raw `.md`), with nicer diagrams and
  sparing interactivity à la `diagrams.html`. Recorded in `docs/dev-loop.md`
  (start/Docs-site-loop/end), `PLAN.md` (parallel track), and here. **Sequencing:**
  build M1 + author `05`/`06` Markdown now; batch-graduate `learnings/01–06` to HTML
  at M1 milestone-doc time. ⏳ **Owed debt:** back-graduate `learnings/01–04`
  (pre-existing) along with the two new ones.
- **Weights fetched:** `model.safetensors` (1.4 GB) downloaded via
  `fetch_model.py --weights` — `fs inspect` will open a real file.
- **Diagram polish:** fixed two misaligned ASCII boxes in `05-reading-shapes.md`
  (the `A·B` inner-dim bracket and the residual-stream block) — miscounted widths,
  not glyph issues. Reinforces the plan to render Learnings diagrams properly in
  HTML so alignment stops being hand-counted.

**Stopped here:** scaffold complete and green; **no helper bodies implemented yet**
(all `todo!()`). `cargo build` + `cargo clippy` pass. Resume at `Mmap::open`.

**Next:** implement bottom-up, one helper at a time (M0 cadence): `Mmap::open`
(the FFI) → `SafeTensors::load` (u64 header len → JSON header → tensor directory) →
`Config::load` → `expected_tensors`/`cross_check` → the `render_*` table, verifying
against the real `model.safetensors` (1.4 GB, already on disk). Then `docs/02-weights.md`.

---

## Session 7 — 2026-06-23 — Future-proof: single-file `tokenizer.json` + special tokens

**Did:** switched the tokenizer's data source from the GPT-2 split `vocab.json` +
`merges.txt` to the official **single `tokenizer.json`** (what newer models ship),
and implemented special tokens off the back of it.
- Replaced the file-reading `load_vocab`/`load_merges` with JSON-parsing helpers:
  `build_vocab` (`model.vocab`), `build_merges` (`model.merges` — an array of
  `["l","r"]` pairs, rank = index; also accepts the legacy `"l r"` string form for
  cross-model robustness), `extract_pattern` (regex from `pre_tokenizer`),
  `build_special_tokens` (`added_tokens`), plus a `read_json` helper.
- **Special tokens are now real:** `special_tokens` (content→id) + `special_ids`
  (id→content); `encode` carves special literals out before BPE via
  `split_on_special_tokens` (longest-match), `decode` emits their literal text.
- Consolidated `BadVocab`/`BadMerges` → one `BadTokenizer { path, message }` (single
  file now). Regex is read from `tokenizer.json`; `PRETOKENIZE_PATTERN` kept as a
  `#[cfg(test)]` reference for the unit tests.
- Trimmed `scripts/fetch_model.py` (no longer fetches `vocab.json`/`merges.txt`) and
  deleted the now-unused local copies from `models/qwen3-0.6b/`.
- Tests: **17 unit** (incl. `build_vocab`/`build_merges` array + legacy forms,
  `extract_pattern`, special-token carving/decoding) + **2 golden integration**
  (the 14-case parity test, plus a special-token test: `<|im_start|>` → 151644,
  carving, round-trip). All green; `cargo clippy` clean.
- Swept all comments/docs/site for the change: `docs/01-tokenizer.md` + `.html`
  (data section, gotchas, verify/deferred), `docs/testing.md`, this file.

**Decisions resolved this session:**
- **Single source of truth = `tokenizer.json`.** Future-proof: newer models ship
  only this file; merges are structured (no whitespace-splitting), and specials +
  regex live inline. ✅
- **`build_merges` accepts both** the modern `["l","r"]` array and legacy `"l r"`
  string forms, for cross-model robustness. ✅
- **Special tokens implemented now**, not deferred. The remaining special-token
  work — the **chat template** (wrapping turns in `<|im_start|>`/`<|im_end|>`) — is
  an M3 concern. ✅
- ⏳ **Still deferred:** the `bpe` HashMap memoization (see Session 5 + the `bpe` doc).

**Next:** M1 — load the weights (safetensors + `config.json`; `fs inspect model/`).

---

## Session 6 — 2026-06-23 — M0 implemented end-to-end; golden passes

**Did:** built the whole tokenizer, bottom-up, one helper at a time (each unit-tested),
then wired and verified it against the real model.
- `build_byte_encoder` — GPT-2 byte→unicode bijection (188 identity + 68 spares from
  `U+0100`). Tested: landmarks (`space→Ġ`, `newline→Ċ`, the `0x7F` gap) + bijection.
- `load_vocab` / `load_merges` — serde_json for vocab → forward map + dense
  id-indexed reverse `Vec` (names the exact defect on a bad vocab); merges → ranks,
  first real merge = rank 0. Tested incl. the rank off-by-one and malformed lines.
- `bpe` + `adjacent_pairs` + `merge_pair` — global-min-rank merge loop; `merge_pair`
  is non-overlapping. `bpe_reproduces_the_hello_trace` runs the `"hello"`→14990 trace
  from Session 5 as an executable test.
- `pretokenize` — Qwen's exact pattern copied verbatim from `tokenizer.json` into
  `PRETOKENIZE_PATTERN` (raw string), via `fancy-regex`. Tests grounded against the
  official pre-tokenizer's output; one asserts chunks tile the whole input.
- Wired `load` / `encode` / `decode`; added `tests/golden_tokenizer.rs` (encode = official
  IDs, decode = official text, round-trip) over all 14 cases. **All pass.** CLI verified
  on `hello world`, the France sentence, and the emoji case. No build warnings.
- Wrote [`docs/01-tokenizer.md`](docs/01-tokenizer.md) — the milestone writeup (pipeline,
  data, the `hello` trace, gotchas, verification), cross-linked to the book / `ds4` /
  Raschka and to [`learnings/03-bpe.md`](docs/learnings/03-bpe.md).

**Decisions / notes resolved this session:**
- **Special tokens:** stubbed as an empty map for M0 (`#[allow(dead_code)]`, "phase 2"
  note). Golden uses `add_special_tokens=False` with no special-token literals, so the
  empty stub is correct. Real carving is the natural M0 follow-up (chat templates). ✅
- **Golden integration test** skips with a notice when `models/` is absent, so
  `cargo test` is green on a fresh checkout but validates fully once assets are fetched. ✅
- **`decode` lossy:** `String::from_utf8_lossy` (an arbitrary id slice can split a
  multi-byte char); a faithful round-trip is always valid UTF-8 anyway. ✅
- ⏳ **Still deferred:** the `bpe` HashMap memoization (see Session 5 + the `bpe` doc).

**Possible next steps (M0 wrap / M1 start):**
1. (optional) Implement special-token carving + a chat-template encode path.
2. (optional) Distill `docs/01-tokenizer.md` into the HTML site + update the sync ledger.
3. **M1 — load the weights:** parse safetensors + `config.json`; `fs inspect model/`.

---

## Session 5 — 2026-06-23 — `bpe` algorithm walkthrough + locked design

**Did:**
- Walked the `bpe` pseudo-code together before writing it. Grounded it in a real
  trace of `"hello"` against Qwen's actual `merges.txt` (ranks = line − 2):
  `[h,e,l,l,o] →(e,l)#45→ [h,el,l,o] →(l,o)#129→ [h,el,lo] →(el,lo)#4535→
  [h,ello] →(h,ello)#14734→ [hello] → id 14990`. The lesson: pick the
  global-min rank each pass (the first merge is in the *middle*, not at `(h,e)`),
  and a high-priority pair can be permanently starved once its members merge.
- Updated `src/tokenizer.rs`: rewrote the `bpe` doc/pseudo-code to the locked
  design and added two private helper sketches, `adjacent_pairs` + `merge_pair`.
  `cargo build` ✓.

**Decisions resolved this session (the `bpe` design):**
- **Final lookup:** `token_to_id.get(s).ok_or(UnknownToken)`, not the panicking
  `token_to_id[s]` index — turn the "impossible" miss into a typed error. ✅
- **Ties:** none possible (ranks are unique line numbers), so no defensive
  tie-break logic. ✅
- **Structure:** `adjacent_pairs` and `merge_pair` are private helpers (testable);
  `merge_pair` consumes pairs non-overlapping left→right, so `[a,a,a]` + `(a,a)`
  → `[aa,a]`. ✅
- **Performance:** keep the naive O(n²)-per-chunk loop for M0 (pretokenize bounds
  n to one word). ⏳ **DEFERRED OPTIMIZATION:** memoize `bpe(word)` in a HashMap
  (GPT-2-style) in a later pass — recorded here + in the `bpe` doc comment. ✅
- **Docs:** the `"hello"`→14990 trace is the canonical worked example for
  `docs/01-tokenizer.md`. ✅

**Next:** implement in PROGRESS order — `build_byte_encoder`, then
`load_vocab`/`load_merges`, then `bpe` (now fully specced) with `adjacent_pairs`/
`merge_pair`, then `pretokenize`, then wire `encode`/`decode` + the golden test.

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
