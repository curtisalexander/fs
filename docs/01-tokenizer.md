# 01 — Tokenizer (M0): text ↔ token IDs

> **Milestone:** M0 · **Status:** ✅ core done — all 14 golden cases pass, CLI runs · **Date:** 2026-06-23
>
> 📖 *Inference Engineering* §2.2 "LLM Inference Mechanics" (p.46)
> 🔧 `ds4`: `ds4.c` — `bpe_tokenize_text` (~`ds4.c:21140`), `gpt2_byte_to_codepoint` (`ds4.c:20896`), the `str_i32_table` vocab + merge-rank hash tables (`ds4.c:20689`)
> 🧭 Raschka, "Build a Large Language Model (From Scratch)" — BPE chapter
> 🔗 concept background: [`learnings/03-bpe.md`](learnings/03-bpe.md) (learned-once vs replayed) · [`learnings/02-radix-tree.md`](learnings/02-radix-tree.md) (why a hash table, not a trie)

The model's front door. M0 turns a string into the integer IDs the model consumes,
and turns IDs back into text — using Qwen3-0.6B's **own** byte-level BPE vocabulary,
so our IDs match the official tokenizer exactly. No GPU, no weights; self-contained.

For *why* BPE works the way it does (training learns the merges; inference just
replays them), read [`learnings/03-bpe.md`](learnings/03-bpe.md) first. This doc is
**what we built** and the gotchas we hit building it.

```
$ fs tokenize "hello world"      →  14990 1879
$ fs detokenize 14990 1879       →  hello world
```

---

## The pipeline

Encoding is four stages; the key structural fact is that they're **nested** — a
coarse split into words, and *within each word*, a fine split into bytes that BPE
merges back up. (This nesting is the thing most people get backwards; see the
gotcha below.)

```
text ──pretokenize──▶ chunks ("words")
                         │  for each chunk, independently:
                         ▼
              bytes→byte-level-unicode  (stage 2)
                         ▼
                   bpe merge loop       (stage 3)
                         ▼
                   piece → id lookup    (stage 4)
                         ▼
                    concatenate ──▶ token IDs
```

| stage | what | code (`src/tokenizer.rs`) |
|---|---|---|
| 1 | split text into word-chunks via Qwen's regex | `pretokenize` + `PRETOKENIZE_PATTERN` |
| 2 | remap each raw byte to a printable char | `build_byte_encoder` |
| 3 | greedily merge adjacent pairs by rank | `bpe` + `adjacent_pairs` + `merge_pair` |
| 4 | look each surviving piece up as an id | the tail of `bpe` |

**Decoding** reverses 4→2: `id → piece` (`id_to_token`), concatenate, then undo the
byte map (`byte_decoder`) to recover the raw UTF-8 bytes. No regex, no merging —
the merges are already baked into the pieces.

---

## The data we load

From `models/qwen3-0.6b/` (fetched by `scripts/fetch_model.py`, git-ignored):

- **`vocab.json`** — `{ "<piece>": id }`, 151,643 entries, ids contiguous `0..=151642`.
  Keys are in *byte-level-unicode*, so `" world"` is stored as `"Ġworld"`.
  → `token_to_id` (forward) + `id_to_token` (a dense `Vec` indexed by id).
- **`merges.txt`** — a `#version` header, then one `left␣right` rule per line in
  **priority order**. → `merge_ranks: (left, right) → rank`, where rank = position
  after the header (first rule = rank 0).
- The **pre-tokenization regex** — copied verbatim out of `tokenizer.json`'s
  `pre_tokenizer` field.

`Tokenizer::load` calls the parsers, builds the byte map and its inverse, and
compiles the regex. Special tokens are stubbed empty for M0 (see below).

---

## The worked example: `"hello"` → `14990`

`"hello"` is one chunk (no spaces), all printable ASCII, so its byte-level form is
just `"hello"`. BPE starts from single chars and applies the lowest-rank merge each
pass (`#N` = the rank from Qwen's real `merges.txt`):

| pass | symbols | candidate pairs (rank) | winner |
|---|---|---|---|
| 1 | `h e l l o` | (h,e)=127, **(e,l)=45**, (l,l)=398, (l,o)=129 | (e,l) |
| 2 | `h el l o` | (h,el)=48866, (el,l)=357, **(l,o)=129** | (l,o) |
| 3 | `h el lo` | (h,el)=48866, **(el,lo)=4535** | (el,lo) |
| 4 | `h ello` | **(h,ello)=14734** | (h,ello) |
| 5 | `hello` | none | — stop |

`token_to_id["hello"]` = **14990**.

Two things this real trace teaches that a toy one wouldn't:

1. **The first merge is in the middle** (`e+l`), not at the front — even though
   `(h,e)` was a valid merge sitting right there. `(e,l)`'s rank 45 beats it.
   *Left-to-right greedy would be wrong.* We must pick the global-minimum rank.
2. **`(h,e)` then never fires.** Once `e` is absorbed into `el`, `h` never sees an
   `e` neighbor again — it waits and eventually merges with the whole `ello`. A
   high-priority pair can be permanently *starved* because its members got eaten by
   even-higher-priority merges. That starvation is BPE behaving correctly.

This trace is pinned as the unit test `bpe_reproduces_the_hello_trace`, so the doc
and the code can't drift.

---

## Gotchas & decisions

- **Pre-tokenize splits into *words*; `bpe` splits each word into *characters*.**
  Both happen, nested. `bpe`'s `piece.chars()` runs on **one chunk**, not the whole
  string, and merges never cross a chunk boundary. The boundaries are sacred: that's
  what makes `"dog"` tokenize the same in `"dog."` and `"dog!"`.

- **A leading space attaches to the following word.** `"hello world"` → `["hello",
  " world"]`, so `" world"` (→ `"Ġworld"`) is a *different* first token than
  `"world"`. The golden encodes this: `"hello world"` → `[14990, 1879]` vs
  `" hello world"` → `[23811, 1879]`. If the byte map or regex is wrong, this is the
  first thing that breaks.

- **The byte map is a bijection over all 256 bytes** (`build_byte_encoder`, the
  GPT-2 construction). 188 "safe" bytes map to themselves; the other 68 take spare
  codepoints `U+0100…`. So space (`0x20`) → `Ġ` (`U+0120`), newline → `Ċ`. Because
  it's a bijection, `byte_decoder` is just the inverse — and decode of any vocab
  piece can never hit an unknown char. Proven by `byte_encoder_is_a_bijection`.

- **Each digit is its own chunk.** Qwen's pattern uses `\p{N}` (one), not `\p{N}+`,
  so `"123"` → `["1","2","3"]`. Easy to get wrong by copying GPT-4's `\p{N}{1,3}`.
  We copy Qwen's pattern *verbatim* from `tokenizer.json` rather than retype it.

- **Merge rank is off-by-one bait.** The first real merge line is rank **0**, not 1
  — the `#version` header is skipped without consuming a rank. A wrong base would
  silently corrupt *every* tokenization. Guarded by `load_merges_ranks_from_zero...`.

- **`merge_pair` is non-overlapping.** Merging `(a,a)` over `[a,a,a]` yields
  `[aa,a]`, not a reused middle `a`. The implementation consumes both halves on a
  hit (a second `.next()`), so the cursor lands past the pair.

- **Lookups return typed errors, never panic.** A piece outside the vocab →
  `UnknownToken` (theoretically impossible, but surfaced not crashed); an
  out-of-range id in `decode` → `InvalidTokenId`. We use `.get().ok_or(...)`, not
  the panicking `[index]`.

---

## Verification — the M0 "done" gate

`tests/golden/tokenizer.json` holds 14 cases generated by the **official** HF
tokenizer (`scripts/gen_golden.py`) with `add_special_tokens=False`: ASCII,
leading/trailing spaces, CJK, emoji, code with tabs/newlines, digit runs.

`tests/golden_tokenizer.rs` asserts, for every case:

1. `encode(text)` == the official IDs,
2. `decode(official_ids)` == the official text,
3. round-trip `decode(encode(text))` == `text`.

All 14 pass. Exact-ID parity across a diverse set *is* the definition of "works
with this model" — it proves the byte map, the regex, and the merge order are all
correct simultaneously. (The integration test skips with a notice if `models/` is
absent, so `cargo test` is green on a fresh checkout; it validates fully once the
assets are fetched.) Plus 13 unit tests cover each helper in isolation.

---

## What's stubbed / deferred

- **Special tokens** (`<|im_start|>`, `<|endoftext|>` …) are stubbed as an empty
  map. They must match verbatim and *bypass* BPE; the golden never exercises them.
  This is the natural M0 follow-up if we want chat-template encoding. (The gap
  between `vocab.json`'s 151,643 and config's `vocab_size` 151,936 is these tokens.)
- **`bpe` performance.** The merge loop rescans all pairs each pass — O(n²) per
  chunk. Fine because `pretokenize` bounds n to one word. **Deferred optimization:**
  memoize `bpe(word)` in a `HashMap` (GPT-2 does this). Recorded in the `bpe` doc
  comment and `PROGRESS.md`.

**Next:** M1 — load the weights (safetensors); parse `config.json` and map every
tensor. See [`PLAN.md`](../PLAN.md).
