# Learning 02 — Radix trees (and what ds4 *actually* uses them for)

> **Date:** 2026-06-14 · **Context:** understanding ds4's data structures · **Status:** clarified a doc error
>
> 📖 *Inference Engineering* §2.2 "LLM Inference Mechanics" (p.46) — tokenization
> 🔧 `ds4`: `rax.c` / `rax.h` (antirez's `rax`), used in `ds4_server.c`; tokenizer hash table in `ds4.c`
> 🧭 antirez's `rax` is the same radix-tree library Redis uses for stream IDs

We went looking at this because our earlier notes said *"ds4's tokenizer uses a
radix tree."* Reading the code, **that turned out to be wrong** — and untangling
it is a good excuse to learn what a radix tree actually is and when you'd reach
for one.

---

## What a radix tree is

A **radix tree** (a.k.a. radix trie, or *compact prefix tree*) stores **strings as
keys**: the path from the root down to a node spells out the key. Its defining
trick is **edge compression**.

**Start from a plain trie** — one character per edge. Storing `"tea"`, `"team"`,
`"ten"`:

```
(root)
  └─ t ── e ── a ── m      "team"
                └ ●        "tea"
          └ n ── ●         "ten"
```

Every node branches up to *N* ways (one per possible next byte). The waste: a long
run with no branching still costs one node per character. `"tokenizer"` would be a
9-node single-file chain — bad for both memory and pointer-chasing.

**The radix tree fix:** collapse any chain of single-child nodes into **one edge
labeled with the whole substring**. Nodes appear *only where keys diverge*:

```
(root)
  └─"te"
      ├─"a" ─────●          "tea"
      │     └─"m"●          "team"
      └─"n" ─────●          "ten"
```

`"te"` is stored once; the tree branches only where the keys actually split (after
`te`: `a` vs `n`; after `tea`: end vs `m`). Edges are labeled with **sequences**,
not single symbols — that's the "radix" idea.

### Why you'd use one

- **Prefix-structured** — keys sharing a prefix share the path. Natural for
  "everything starting with `foo`", longest-prefix matching, ordered iteration.
- **Memory-efficient vs a plain trie** — no long single-child chains.
- **Lookup is O(key length)**, independent of how many keys are stored — you walk
  the query's bytes down the tree, never comparing against the other keys.

### How it stacks up against the alternatives

| | Hash table | Radix tree | Sorted array + binary search |
|---|---|---|---|
| Exact lookup | ~O(1) after hashing | O(key length) walk | O(key length × log n) |
| Ordered iteration | ✗ | ✓ (lexicographic) | ✓ |
| Prefix / range queries | ✗ | ✓ (the whole point) | partial |
| Shares memory across common prefixes | ✗ | ✓ | ✗ |
| Cache behavior | one hash, random probe | many small pointer hops | good (contiguous) |

Headline: a **hash table** wins for pure "exact key → value" when you never need
order or prefixes. A **radix tree** wins when **prefix relationships or sorted
order matter**.

---

## What ds4 actually does

Two separate data structures, two separate jobs — and the radix tree is **not** the
one in the tokenizer.

### The tokenizer uses a hash table

ds4's byte-level BPE does repeated **exact** lookups: "is this exact byte string a
known token, and what's its id?" and "what's the merge rank of this exact pair?"
No prefixes, no ordering — so it uses an open-addressing hash table:

- `str_i32_table` — `ds4.c:20689` (power-of-two capacity, `hash_bytes` + linear
  probing).
- Vocab fields `token_to_id` and `merge_rank` — `ds4.c:20791`.
- Lookups via `table_get` in the BPE inner loop — e.g. `ds4.c:21016`.

### The radix tree lives in the server's agent memory

antirez's `rax` library *is* compiled into ds4, but it's used by the built-in
agent's **tool-memory store**, not the tokenizer:

- `m->by_id = raxNew();` and `m->by_block = raxNew();` — `ds4_server.c:7764`.
- It maps string IDs and "dsml" block content to memory entries
  (`raxInsert` / `raxFind` around `ds4_server.c:7808`).

That's a sensible fit: memories are keyed by strings you want **ordered and
prefix-addressable** access to — exactly the radix tree's strength. (Redis, also
antirez, uses the same `rax` for stream IDs for the same reasons.)

---

## Mental model to keep

> **Exact key → value, nothing more?** Hash table.
> **Need prefixes, longest-match, or sorted order?** Radix tree.
>
> ds4's tokenizer is the first case (hash table, `ds4.c`); ds4's agent memory is
> the second (radix tree, `ds4_server.c`). A radix tree *can* tokenize (handy for
> greedy longest-match schemes like WordPiece/Unigram), but byte-level BPE doesn't
> need it.

### For our M0 tokenizer

When we build the tokenizer in Rust at M0, follow ds4's actual design: a hash map
(`HashMap<Vec<u8>, u32>` for token→id, plus merge ranks) is all byte-level BPE
needs. Reach for a trie/radix structure only if we later add a longest-match
tokenizer that benefits from prefix walking.
