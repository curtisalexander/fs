# Learning 08 — Row-major layout & strides: from a shape to a byte offset

> **Date:** _stub — to write at M2_ · **Context:** M2, indexing into the tensor blob · **Status:** 🌱 stub
>
> 📖 *Inference Engineering* §2.1 (the transformer block, p.42) — tensors in memory
> 🔧 `ds4`: `metal/dense.metal` (matmul indexing), `metal/get_rows.metal` (row gather)
> 🧭 NumPy's ["internal memory layout of an ndarray"](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray) — strides, C- vs F-order

> **STUB — do not distill to HTML yet.** Placeholder so M2 doesn't miss this
> teaching moment. [`learning 01`](01-safetensors-vs-gguf.md) already notes the
> safetensors blob is "row-major (C-contiguous)" *in passing*; this note is where
> we actually explain what that means and turn it into indexing code. Flesh it out
> when M2 first reads weights element-by-element (embedding gather → matmul).

---

## What this note will cover (outline)

- **A shape is not a layout.** `[2048, 1024]` tells you the axes; it does *not* by
  itself tell you which byte a given `(row, col)` lives at. That's what the
  **layout** (row-major vs column-major) plus the **strides** encode.
- **Row-major / C-contiguous.** The last axis is contiguous: element `(i, j)` of an
  `[R, C]` tensor sits at flat index `i·C + j`. Draw the `[out, in]` weight this
  way and connect it back to the `[out, in]` convention from
  [`learning 05`](05-reading-shapes.md) — a "row" of the weight is one output
  feature's `in`-wide vector, laid out contiguously.
- **Strides = the multipliers.** For `[R, C]` row-major, `strides = [C, 1]`
  (in elements). Offset(`i, j`) = `i·stride0 + j·stride1`. Generalize to N-D as a
  dot product of index and strides. Why strides make transpose/slice **free** (you
  change the strides, not the bytes) — and why we *won't* lean on that in M2's slow
  first pass (clarity over cleverness).
- **byte offset, end to end.** Combine with [`learning 07`](07-bf16.md): a bf16
  `[R, C]` tensor's element `(i, j)` is at **byte** `data_start + t.start +
  (i·C + j)·2`, and you widen those 2 bytes with `bf16_to_f32`. This is the exact
  bridge from "a slice of the mmap" to "a number in a matmul."
- **The gotcha to burn in.** Row-major + the `[out, in]` storage convention is why
  `y = x·Wᵀ` reads *rows* of `W` (each an `in`-wide contiguous run) — the memory
  layout and the transpose line up so the hot loop is cache-friendly. Foreshadow
  the M6 speed lesson (contiguity → coalesced Metal reads).
- **Assert it.** A `[R, C]` tensor must occupy exactly `R·C·dtype.size()` bytes;
  make that check visible (ties back to the M1 `BadTensorInfo` validation).

## When to write this

At M2, the moment we first index a weight: the **embedding gather** (`get_rows` —
pick token id `t`'s row out of `embed_tokens [V, H]`) is the cleanest first
example, then the **matmul** (`dense`) generalizes it. Pull the worked byte-offset
example from real Qwen3-0.6B numbers, mirroring how `learning 05` used real dims.

---

## Cross-links (to wire up when written)

- ⬅ [`learning 05 · reading shapes`](05-reading-shapes.md) — the `[out, in]`
  convention this note turns into byte offsets.
- ⬅ [`learning 07 · bf16`](07-bf16.md) — the 2-bytes-per-element the strides step
  over.
- ⬅ [`learning 01 · safetensors vs GGUF`](01-safetensors-vs-gguf.md) — "row-major
  (C-contiguous)", stated there, explained here.
- 🔧 `ds4`: `metal/get_rows.metal`, `metal/dense.metal`.
