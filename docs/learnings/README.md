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
| 05 | [reading shapes](05-reading-shapes.md) | 2026-06-24 | how dimensions line up; `[out,in]`, the residual stream, head_dim decoupling, GQA |
| 06 | [mmap (raw POSIX FFI)](06-mmap.md) | 2026-06-24 | turning a file into memory zero-copy; the `mmap`/`munmap` FFI + RAII wrapper |
| 07 | [bf16](07-bf16.md) | 2026-07-06 | the weights' number format; bf16 = fp32's top 16 bits; why we widen lazily |
| 08 | [row-major & strides](08-row-major-strides.md) 🌱 | _M2_ | **stub** — shape→byte-offset; row-major layout, strides, indexing the blob |
