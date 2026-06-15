# Learning 03 — Byte-pair encoding: learned once, replayed forever

> **Date:** 2026-06-14 · **Context:** before building the M0 tokenizer · **Status:** understood
>
> 📖 *Inference Engineering* §2.2 "LLM Inference Mechanics" (p.46)
> 🔧 `ds4`: `ds4.c` — `bpe_tokenize_text` (~`ds4.c:21140`), `bpe_emit_piece` (~`ds4.c:20965`), `str_i32_table` vocab/merge-rank hash tables (`ds4.c:20689`)
> 🧭 Raschka, "Build a Large Language Model (From Scratch)" — clean BPE walkthrough
> 🔗 see also [`02-radix-tree.md`](02-radix-tree.md) (why the tokenizer is a hash table, not a trie)

We kept saying BPE "keeps merging common adjacent pairs." That phrase hides the
single most important fact about tokenizers: **there are two phases, and only one
of them involves learning.**

---

## The two phases

### Phase 1 — Training the tokenizer *(done once, by the model creators — the "learned" part)*

This is where frequency-driven merging happens. It's unsupervised machine learning
over a big corpus:

1. Vocabulary starts as the **256 raw bytes**.
2. Scan the corpus, count every **adjacent pair** of symbols.
3. Take the **single most frequent pair** (say `t` + `h`), **merge** it into a new
   symbol `th`, give it the next free token id, and append the rule
   `("t","h") → "th"` to an **ordered merges list**.
4. Repeat — re-count (the new symbol can now participate), merge the next most
   frequent pair — until the vocabulary hits a target size (Qwen3 ≈ **151,936**).

The output is two static artifacts that ship *with the model*:

- a **vocab** — token string → id
- an **ordered merges list** — the merge rules, *in the order they were learned*.
  Order = frequency rank, and rank is everything.

### Phase 2 — Encoding text at inference *(what we build at M0 — NOT learned)*

Our tokenizer does **zero learning and zero frequency counting**. It deterministically
**replays** the frozen rules:

1. **Pre-split** the text into coarse chunks with a fixed regex (see
   *The pre-tokenization regex* below). Merges never cross a chunk boundary.
2. Turn each chunk into per-byte symbols: encode it as **UTF-8 bytes**, then map
   each byte through the byte↔unicode table (see *What "raw bytes" means* below).
3. Within each chunk, repeatedly apply the adjacent pair whose merge rule has the
   **lowest rank** (earliest in the merges list = most frequent at training time).
4. Stop when no adjacent pair in the chunk appears in the merges list.

```
"low"  →  l o w            (raw bytes / chars)
          ^merge rank(l,o)=5, rank(o,w)=2  → apply the lower rank first
       →  l ow             (applied o+w)
       →  low              (if (l,ow) is also a rule)
```

That's why ds4 stores a `merge_rank` **hash table** (`ds4.c:20791`): "given this
pair, what's its rule rank?" — pure exact lookup, no counting. The learning already
happened upstream.

> **One-liner:** the model creators *learned* the merges from a corpus; we *replay*
> their frozen merges in rank order. Encoding is deterministic table lookup.

---

## What "raw bytes" means — the encoding question

"Split into raw bytes" hides an assumption: **the input is UTF-8.** There is no
encoding-detection step. The pipeline treats the text as a UTF-8 byte sequence and
walks it one byte (0–255) at a time. ds4 does exactly this — `byte_encode`
(`ds4.c:20914`) iterates the input as raw `uint8_t` bytes.

This matters because **the model's merges were trained over UTF-8 bytes**. Feed the
same text as Latin-1 or UTF-16 and you get different bytes → different merges →
wrong IDs. UTF-8 is part of the contract with the model, not a free choice. (In
Rust this is handled for us: `&str` is *always* valid UTF-8, so `.as_bytes()` is the
right bytes for free.)

The nice consequence: a character is **its UTF-8 bytes**, not one symbol. `é`
(U+00E9) enters BPE as 2 bytes, `中` as 3, an emoji as 4. Because all 256 byte
values are in the base vocab, **anything is representable — no out-of-vocabulary
case, no `UNK` token.** That's the whole reason byte-level BPE exists.

One subtlety: you don't merge over raw `0x00–0xFF` directly. Each byte is first
remapped to a *printable* codepoint via the GPT-2 byte↔unicode table
(`gpt2_byte_to_codepoint`, `ds4.c:20896`): printable ASCII/Latin-1 stay as-is; the
~68 awkward bytes (control chars, space, newline) map to codepoints from 256 up —
which is why a space shows as `Ġ` in vocab dumps. Each remapped codepoint still
stands for exactly one original byte; it just keeps the merge symbols printable and
whitespace-free.

---

## The pre-tokenization regex — yes, really a regex

Before BPE runs, the text is split into coarse chunks by a **single fixed regex**.
The classic GPT-2 pattern:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
```

Read it as: contractions · *optional-space + letters* · *optional-space + digits* ·
*optional-space + punctuation* · *whitespace*. So `"  hello world!"` pre-splits into
`["  hello", " world", "!"]`. Qwen uses a more elaborate cl100k/GPT-4-style variant,
same idea.

**The regex is *not* the tokenizer.** It only pre-splits so the learned merge loop
has clean boundaries — merges never cross a chunk. Without it, BPE would learn
merges across word/space/punctuation boundaries (`"dog."`, `"the quick"`), wasting
vocab and making segmentation wildly context-dependent. The regex is a guard rail
around the real (learned) algorithm.

**Where the regex comes from depends on the model's format:**

- **HuggingFace `tokenizer.json`:** the literal regex string is in the
  `pre_tokenizer` field. You read it out verbatim.
- **GGUF:** stores only a **name** — `tokenizer.ggml.pre = "qwen2"` (or `"llama-bpe"`,
  `"gpt-2"`, …) — *not* the pattern. The engine must already know the regex that name
  refers to. That's why ds4 hardcodes per-model pre-tokenizers, including the
  `"joyai-llm"` special-case (`ds4.c:21121`).

Either way the regex is part of the **model's definition**, not something we invent.

### Isn't this just how a compiler tokenizes?

Largely yes — lexical analysis is *the* textbook use of regular expressions. A
compiler's lexer is specified as regexes (one per token type), compiled to a DFA;
`lex`/`flex` generate exactly that. But the parallel breaks in ways worth holding
onto:

| | Compiler lexer | LLM BPE tokenizer |
|---|---|---|
| Rules are | hand-**designed** in the language spec | **learned** from a corpus by frequency |
| A token is | a grammatical category (`IDENT`, `IF`, `+`) — dozens | a statistical fragment (`" dog"`, `"tion"`) — ~150k |
| The regex | **is** the whole tokenizer | only **pre-splits**; a learned merge loop does the real work |
| Greedy rule | longest-match + rule priority | apply the **lowest-rank merge** |

Deeper: a lexer is **lossless and meaningful** (every token is one lexical unit; the
parser depends on it; there's a *right* answer). BPE segmentation is **arbitrary** —
the same word tokenizes differently with a leading space, and the model is fine
because it trained on that messiness. Fancier schemes exist (Unigram/SentencePiece
uses Viterbi for the most-likely split; WordPiece uses greedy longest-match), and
some research drops tokenization entirely (byte-level models like ByT5). BPE's
greedy-merge is one of the *simpler* methods; the regex pre-tokenizer is widely seen
as an inelegant-but-necessary wart.

---

## Making our tokenizer match a specific model

A tokenizer is correct **only relative to one model's vocab + merges**. "Implementing
BPE" is not enough — we must reproduce *that model's* exact rules and conventions.
For our target (Qwen3, GPT-2-style byte-level BPE):

1. **Load the model's own vocab + merges — never invent them.** From the GGUF
   (`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`) or HuggingFace `tokenizer.json`.
   Same source the model trained with.
2. **Match the byte-level mapping.** Byte-level BPE maps raw bytes through a fixed
   bytes↔unicode table so every byte is a printable char (that's why a space shows
   up as `Ġ` in vocab dumps). Get this table exactly right or nothing lines up.
3. **Match the pre-tokenization regex.** Text is split into pieces by a regex
   (spaces, contractions, digit runs, punctuation) *before* BPE runs. This is the
   most common source of subtle mismatches — ds4 special-cases it per model
   (`bpe_tokenize_text`, ~`ds4.c:21140`). Use **Qwen's** pattern.
4. **Match special-token handling.** `<|im_start|>`, `<|endoftext|>`, etc. are
   inserted as whole tokens and **bypass BPE**.
5. **Verify with round-trip + exact-ID parity** (already the M0 plan, `PLAN.md`):
   run the official HuggingFace tokenizer on a diverse string set — ASCII,
   multilingual, emoji, code with leading whitespace, the special tokens — and
   assert our IDs match **exactly**. `encode` → `decode` must also reproduce the
   input byte-for-byte. Exact-ID parity across a diverse set *is* the definition of
   "works with this model": it proves the byte map, the regex, and the merge order
   are all right.

---

## Mental model to keep

> **Training** = learn merges by frequency over a corpus → frozen `(vocab, merges)`.
> **Inference** = replay merges in rank order. Deterministic, no learning.
> **Correct** = byte-for-byte ID parity with the model's *own* official tokenizer.

### For M0 (Rust)

- `HashMap<Vec<u8>, u32>` for token→id, plus a pair→rank map for merges (mirrors
  ds4's two `str_i32_table`s). No trie needed — byte-level BPE is exact lookups.
- Implement: byte↔unicode table → pre-tokenization regex → greedy lowest-rank merge
  loop → special-token splice.
- Build the parity test harness *first*; treat any single ID mismatch as a bug in
  one of the four conventions above, not a rounding detail.
