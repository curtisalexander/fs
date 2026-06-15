# Learnings

Bite-sized notes on things we figured out along the way — the *why* behind
decisions, and background worth keeping. Each entry cross-links the three sources
(the book, `ds4`, Raschka). These complement the milestone docs (`docs/0X-*.md`)
and the big-picture [`../00-map.md`](../00-map.md).

| # | Note | When | Topic |
|---|---|---|---|
| 01 | [safetensors vs GGUF](01-safetensors-vs-gguf.md) | 2026-06-13 | model file formats; why we go safetensors→GGUF |
| 02 | [radix trees](02-radix-tree.md) | 2026-06-14 | what a radix tree is; ds4's tokenizer is a hash table, not a trie |
| 03 | [byte-pair encoding](03-bpe.md) | 2026-06-14 | BPE: learned once (training) vs replayed (inference); matching a model |
| 04 | [embedding model ≠ token table](04-embedding-models.md) | 2026-06-14 | the three senses of "embedding"; why an embedding model is a forward pass, not a lookup |
