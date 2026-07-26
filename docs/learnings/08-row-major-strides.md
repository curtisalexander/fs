# Learning 08 — Row-major layout & strides: from a shape to a byte offset

> **Date:** 2026-07-26 · **Context:** M2, embedding gather + matmul · **Status:** living
>
> 📖 *Inference Engineering* §2.1 (the transformer block, p.42) — tensors in memory
> 🔧 `ds4`: `metal/dense.metal` (matmul indexing), `metal/get_rows.metal` (row gather)
> 🧭 NumPy's ["internal memory layout of an ndarray"](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray) — strides, C- vs F-order

[`Learning 05`](05-reading-shapes.md) taught us to read a weight shape like
`[out, in]`. That tells us what the axes *mean*. It still does not tell us where
element `(row, col)` lives in memory. M2 finally forces the second question:

> **A shape names the axes. A layout orders their elements. Strides turn an index
> on those axes into one flat offset.**

Our M2 `Matrix` chooses one deliberately simple answer: contiguous, row-major
f32, always. No views, no hidden transpose, no variable strides. The arithmetic
stays visible while we learn it; more flexible layouts can wait until they solve
a measured problem.

---

## A shape is not a layout

Take a toy matrix with shape `[3, 4]`:

```text
                 col 0  col 1  col 2  col 3
row 0              a      b      c      d
row 1              e      f      g      h
row 2              i      j      k      l
```

The shape says there are three rows and four columns. It does **not** say whether
memory stores `a,b,c,d,e,…` (row-major) or `a,e,i,b,…` (column-major). The same
logical grid admits both. Safetensors uses C-contiguous tensors, and our `Matrix`
keeps that row-major layout after widening to f32:

```text
flat Vec<f32>
index:   0  1  2  3 | 4  5  6  7 | 8  9 10 11
value:   a  b  c  d | e  f  g  h | i  j  k  l
         └── row 0 ─┘ └── row 1 ─┘ └── row 2 ─┘
```

The last axis is contiguous: moving one column right moves one element in memory.
Moving one row down skips a complete row of `C` elements. Therefore:

```text
flat_index(row, col) = row · C + col
```

For `(row=2, col=1)` in `[3,4]`: `2·4 + 1 = 9`, which is `j`.

That exact formula appears in `Matrix::row`:

```rust
&self.data[r * self.cols .. (r + 1) * self.cols]
```

The slice is not merely convenient Rust. It makes the invariant concrete: one
logical row is one uninterrupted run of `cols` f32 values.

## Strides are the multipliers

A **stride** says how many elements to skip when an index advances by one along
an axis. A contiguous row-major `[R,C]` matrix has element strides:

```text
axis:       row   col
stride:      C     1

offset(row, col) = row·C + col·1
```

For an N-dimensional row-major shape `[D₀,D₁,…,Dₙ]`, the last stride is `1` and
each stride to its left is the product of all dimensions to its right. The flat
offset is the dot product of indices and strides:

```text
offset(i₀,…,iₙ) = i₀·stride₀ + … + iₙ·strideₙ
```

This is why a tensor library can make transpose "free": swap shape and strides,
leave the bytes alone. A `[R,C]` row-major view has strides `[C,1]`; its transpose
view can report shape `[C,R]` and strides `[1,C]`. But its rows are no longer
contiguous.

M2 intentionally does **not** carry those variable strides. `Matrix` means one
thing: `data.len() == rows·cols`, stride `(cols,1)`. If a later fast kernel wants a
different physical order, making that reorder an explicit copy lets us measure
its cost and payoff rather than hiding it inside a view.

## From the mmap to one Qwen weight

The safetensors reader gives us a tensor-relative byte range. Combine its start
with row-major indexing and [`Learning 07`](07-bf16.md)'s two bytes per bf16:

```text
file_byte(row, col)
  = data_start + tensor.start + (row·C + col)·2
```

Use the real embedding table `[V,H] = [151936,1024]`. The oracle prompt starts
with token id `785` (`"The"`), so embedding gather selects row 785. Relative to
the start of `embed_tokens`:

```text
first element (785,0):       (785·1024 + 0)    · 2 = 1,607,680 bytes
last  element (785,1023):    (785·1024 + 1023) · 2 = 1,609,726 bytes
whole row byte range:        [1,607,680, 1,609,728) = 2,048 bytes
```

Those 2,048 bytes are 1,024 adjacent bf16 values. M2 widens each with
`bf16_to_f32`, and the resulting row occupies 4,096 adjacent bytes in the owned
f32 `Matrix`. The representation changed width; the row-major order did not.

This also explains the M1 validation:

```text
tensor bytes must equal R · C · dtype.size()
```

If that equality fails, no stride formula can make the header and blob agree, so
`SafeTensors::load` rejects the tensor before M2 can index it.

## Gather: copy one contiguous row

Embedding is not multiplication. Given `embed[V,H]` and `ids[seq]`, it copies one
`H`-wide table row per token:

```text
ids [2, 0, 2]       embed [V=4,H=3]          output [seq=3,H=3]
                    row 0: [ 0, 1, 2]         [20,21,22]
       2 ─────────▶ row 2: [20,21,22]         [ 0, 1, 2]
       0 ─────────▶ row 0: [ 0, 1, 2]         [20,21,22]
       2 ─────────▶ row 2: [20,21,22]
```

`embedding_gather` uses `row(id)` and `copy_from_slice`, preserving token order
and asserting every id is below `V`. The operation is the CPU version of `ds4`'s
`get_rows`: compute the row start, then copy a contiguous width.

## Matmul: three indices, three row-major offsets

For `A[M,K] · B[K,N] → C[M,N]`, each output cell is a dot product:

```text
C[i,j] = Σₖ A[i,k] · B[k,j]
```

Row-major turns the three logical indices into flat offsets:

```text
A[i,k] → A.data[i·K + k]
B[k,j] → B.data[k·N + j]
C[i,j] → C.data[i·N + j]
```

That is exactly M2's deliberately naive `(i,j,k)` triple loop. It is not fast:
the `A` walk is contiguous while the `B` walk jumps by `N`. That is useful to
*see*. Tiling, transposition, SIMD, and Metal exist to improve those access
patterns later; changing them now would hide the baseline we need to understand
and measure.

## Linear: why `[out,in]` is good storage

Neural-network weights are stored `[out,in]`, while the math is `Y = X·Wᵀ`:

```text
X[seq,in] · Wᵀ[in,out] → Y[seq,out]

Y[t,o] = Σᵢ X[t,i] · W[o,i]
```

There is no reason to physically transpose `W`. `X.row(t)` and `W.row(o)` are
both contiguous `in`-wide slices, so `linear` dots those two rows directly:

```text
X row t:  [x₀ x₁ x₂ … xᵢₙ₋₁]   contiguous
W row o:  [w₀ w₁ w₂ … wᵢₙ₋₁]   contiguous
             multiply + sum
Y[t,o] = x₀w₀ + x₁w₁ + …
```

The apparent transpose in the equation and the physical `[out,in]` layout fit
together. One weight row contains every incoming weight for one output feature.
This is the bridge from Learning 05's shape convention to executable code.

## The rest of the numerical foundation keeps rows explicit

The same discipline carries through the other helpers implemented beside gather
and matmul:

- **RMSNorm** normalizes one complete vector, and `rms_norm_rows` applies it to
  each `[seq,H]` row without mixing tokens.
- **SiLU** is element-wise, so layout does not change its result.
- **RoPE** rotates two halves of each `d`-wide query/key row. Its `[seq,d]`
  cosine/sine tables are themselves row-major: one contiguous frequency row per
  position. The test locks HF's fp32 table values at Qwen's real final position
  `40959`, where silently computing the table in f64 would drift beyond tolerance.
- **top-k** consumes the final flat `[V]` logit row and preserves token id as the
  original flat index while sorting scores.

Every helper asserts its shape contract before indexing. The known-answer tests
check both the numbers and the loud-failure paths; RoPE additionally checks that
a rotation preserves vector length.

---

## Mental model to keep

> **Shape tells you what an index means. Strides tell you how far it moves.
> Row-major `[R,C]` means strides `[C,1]`, so `(r,c)` is `r·C+c`.**

Then remember the model-specific payoff: `[out,in]` stores one output neuron's
weights as one contiguous row, so `x·Wᵀ` needs no physical transpose.

---

## Cross-links

- ⬅ [`Learning 05 · reading shapes`](05-reading-shapes.md) — the `[out,in]`
  convention this note turns into offsets and row dots.
- ⬅ [`Learning 07 · bf16`](07-bf16.md) — the two-byte source values those offsets
  step over before M2 widens them.
- ⬅ [`Learning 01 · safetensors vs GGUF`](01-safetensors-vs-gguf.md) — the tensor
  blob and C-contiguous premise.
- 🔗 [`Learning 10 · transformer block anatomy`](10-transformer-block-anatomy.md)
  — where linear, RMSNorm, SiLU, and RoPE fit in the architecture.
- 🔧 `src/tensor.rs` — `Matrix`, `matmul`, and `[out,in]` `linear`.
- 🔧 `src/forward.rs` — embedding gather, RMSNorm, SiLU, RoPE, and top-k.
- 🔧 `ds4`: `metal/get_rows.metal`, `metal/dense.metal`, `metal/norm.metal`,
  `metal/dsv4_rope.metal`.
