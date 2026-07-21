# M1 — Load the weights

> **Status:** ✅ done · **Artifact:** `fs inspect <model_dir>` · **Verify:** the
> real Qwen3-0.6B cross-checks clean (311 tensors) against its own `config.json`
>
> 📖 *Inference Engineering* §4.2.2 "Model File Formats" (p.103) · §2.2.1 (p.49)
> 🔧 `ds4`: `ds4.c` owns GGUF loading (mmap-based) + `gguf-tools/`
> 🧭 Raschka's workflow: technical report → HF config → reference implementation

M0 turned text into token IDs with **no weights at all**. M1 is where the 1.4 GB
pile of numbers finally shows up — and the whole milestone is one question:

> **Does this file contain exactly the tensors, at exactly the shapes, that the
> architecture implies?**

`fs inspect models/qwen3-0.6b` answers it — loading `config.json` + `model.safetensors`,
deriving the expected tensor set from the config, cross-checking it against the file,
and printing a shape-first report. No forward pass yet (that's M2); this is the
*handshake* that makes M2 safe to write.

---

## A model is two files, and you need **both**

This is the weights-side echo of [learning 09](learnings/09-config.md), which made
the point from the config side. Restated here because M1 is where it becomes concrete:

- **`model.safetensors` alone** is a flat blob — 596M numbers with no grammar.
  Nothing in it says a `[2048,1024]` matrix is `q_proj`, that there are 28 blocks,
  or where one layer ends.
- **`config.json` alone** is a recipe with no ingredients — the *shape* of a network
  with no learned values.

`fs inspect` is literally this handshake made executable: it proves the tensors in
the `.safetensors` file are exactly the ones the config implies, at the shapes it
implies. **A checksum can't do that** — only the config says what "correct" means.
See [learning 10 §1](learnings/10-transformer-block-anatomy.md) for where that
"correct" comes from (the reference implementation, not memory).

---

## How loading works, bottom-up

We built it as a stack of small, separately-tested helpers — the M0 cadence, one
helper at a time.

| helper | file | what it does | learning |
|---|---|---|---|
| `Mmap::open` | `src/safetensors.rs` | maps the whole file zero-copy via **raw POSIX `mmap` FFI** (no `libc` crate); RAII `munmap` on `Drop` | [06 · mmap](learnings/06-mmap.md) |
| `SafeTensors::load` | `src/safetensors.rs` | reads `[u64 len][JSON header][blob]` into a validated `Tensor` directory | [01 · formats](learnings/01-safetensors-vs-gguf.md) |
| `Config::load` | `src/config.rs` | parses `config.json` → the 7 named dims + M2/M3 scalars (no silent defaults) | [09 · config](learnings/09-config.md) |
| `expected_tensors` | `src/inspect.rs` | the config-derived spec: 3 global + 11×L, as code | [10 · anatomy](learnings/10-transformer-block-anatomy.md) |
| `cross_check` | `src/inspect.rs` | diffs the spec against the file; totals params | [05 · shapes](learnings/05-reading-shapes.md) |
| `render_*` + `run` | `src/inspect.rs` | the shape-first legend / table / verdict | [05 · shapes](learnings/05-reading-shapes.md) |

### The safetensors format (three regions)

```text
[ 8 bytes: u64 LE = header length N ][ N bytes: JSON header ][ raw tensor blob ]
```

The JSON header maps each name → `{dtype, shape, data_offsets:[s,e]}`, where `[s,e)`
indexes the trailing blob. So "reading" the file is: read 8 bytes, parse N bytes of
JSON, and every tensor is now a `[s,e)` slice of the rest. We **never copy a weight**
— tensors stay as borrowed byte slices into the mapping. `parse_tensor_entry` earns
its keep with two checks — `end ≤ blob_len` and `end - start == shape·dtype.size()`
— so a self-inconsistent header fails loudly at load, not as a mis-slice in M2.

### Two decisions worth naming

- **mmap, not read.** The OS maps the file into our address space and pages it in
  lazily; the 1.4 GB never lands on our heap. Done with raw FFI to match the
  no-hidden-abstraction ethos — see [learning 06](learnings/06-mmap.md).
- **bf16 stays lazy.** Tensors keep their raw bf16 bytes; a `bf16_to_f32` helper
  exists but is only *called* in M2. Eager conversion would copy 1.4 GB → ~2.8 GB of
  f32 and defeat the mapping. See [learning 07](learnings/07-bf16.md).

---

## The cross-check: deriving what *should* be there

`expected_tensors(cfg)` is [learning 05](learnings/05-reading-shapes.md)'s shape spec
written as code — the same set [learning 10](learnings/10-transformer-block-anatomy.md)
walks tensor by tensor. For Qwen3-0.6B it emits **3 global + 11 per layer × 28 = 311**:

- **global:** `embed_tokens [V,H]`, `model.norm [H]`, `lm_head [V,H]`
- **per block:** `input_layernorm`, `q/k/v_proj` (the GQA asymmetry: `[2048,1024]` vs
  `[1024,1024]`), `q/k_norm [d]`, `o_proj [H,q]`, `post_attention_layernorm`, and
  SwiGLU's `gate/up [I,H]` + `down [H,I]`.

`cross_check` then walks that spec against the file in three passes: required-missing
and shape-mismatch → **problems** (each naming the offending dim); any file tensor not
in the spec → an **extra** (also a problem); and a params total. A clean result is the
M1 verification — and those asserts carry into M2, so a mis-wired matmul fails
*loudly* instead of producing quiet garbage.

### The gotcha the file taught us: tied ≠ absent

The plan assumed `tie_word_embeddings: true` ⟹ no `lm_head.weight` in the file.
**The real file has one** — and its bytes are *byte-for-byte identical* to
`embed_tokens`. "Tied" is a statement about the **math**, not the **file**: a tied
export may omit `lm_head` *or* ship a redundant copy (Qwen3-0.6B does the latter).
Had we coded from the assumption, the cross-check would have flagged a real tensor as
an "extra" and reported a false failure.

So `expected_tensors` marks `lm_head` **optional when tied**, and `cross_check`
reports two param counts:

| | count | meaning |
|---|---:|---|
| **stored** | 751,632,384 | the naive sum over every file tensor (the table stored *twice*) |
| **logical** | 596,049,920 | the redundant copy deduped — the "**0.6B**" |
| embeddings | 155,582,464 | **26.1%** of logical — a quarter of the model is the vocabulary |

The full story — and *why we read the header instead of trusting memory* — is
[learning 10 §3](learnings/10-transformer-block-anatomy.md).

---

## What `fs inspect` prints

```text
── dimensions (from config.json) ───────────────────────────────────────────
  V  vocab_size             151936   distinct tokens
  H  hidden_size              1024   residual-stream width (the bus)
  L  num_hidden_layers          28   transformer blocks
  d  head_dim                  128   width of one attention head
     num_attention_heads        16   query heads → q width = 16·128 = 2048
     num_key_value_heads         8   kv heads → kv width = 8·128 = 1024   (GQA group 2)
  I  intermediate_size        3072   FFN inner width
  weights are stored [out, in]; read a row as   in ──▶ out   (y = x·Wᵀ)

── tensors ─────────────────────────────────────────────────────────────────
  TENSOR                          DTYPE SHAPE            PARAMS   in ──▶ out
  global
    model.embed_tokens.weight     BF16  [151936, 1024]  155,582,464  id ──▶ H   (row gather)
  each block  × 28   (shown: layer 0)
    input_layernorm.weight        BF16  [1024]                1,024  scale 1024
    self_attn.q_proj.weight       BF16  [2048, 1024]      2,097,152  1024 ──▶ 2048
    self_attn.k_proj.weight       BF16  [1024, 1024]      1,048,576  1024 ──▶ 1024
    self_attn.v_proj.weight       BF16  [1024, 1024]      1,048,576  1024 ──▶ 1024
    self_attn.q_norm.weight       BF16  [128]                   128  scale 128
    self_attn.k_norm.weight       BF16  [128]                   128  scale 128
    self_attn.o_proj.weight       BF16  [1024, 2048]      2,097,152  2048 ──▶ 1024
    post_attention_layernorm.weight BF16 [1024]                1,024  scale 1024
    mlp.gate_proj.weight          BF16  [3072, 1024]      3,145,728  1024 ──▶ 3072
    mlp.up_proj.weight            BF16  [3072, 1024]      3,145,728  1024 ──▶ 3072
    mlp.down_proj.weight          BF16  [1024, 3072]      3,145,728  3072 ──▶ 1024
  final
    model.norm.weight             BF16  [1024]                1,024  scale 1024
    lm_head.weight   (tied)       BF16  [151936, 1024]  155,582,464  1024 ──▶ 151936

── verdict ─────────────────────────────────────────────────────────────────
  ✓ all 311 expected tensors present, shapes match the config
  note: lm_head.weight present but tied — a redundant byte-identical copy of embed_tokens (155,582,464 params counted once)
  params: 751,632,384 stored · 596,049,920 logical (the "0.6B")
  embeddings: 155,582,464 = 26.1% of logical
```

Every number is **derived** — the GQA asymmetry, the deduped 596M, the 26.1% —
nothing hard-coded. A shape mismatch makes `run` return not-clean, which the CLI
turns into a non-zero exit code (`fs inspect` can gate a build).

---

## How we know it's right

**51 unit + 2 golden integration tests; `cargo clippy --all-targets` clean.** The
verification is layered:

- **Per-helper units** hit reality: `Mmap::open` round-trips raw bytes through the
  live syscall; `SafeTensors::load` parses the real 1.4 GB header; `Config::load`
  reads the shipped `config.json`; `expected_tensors` derives exactly 311 from it.
- **`cross_check` synthetic cases** (built through the real `SafeTensors::load`)
  cover clean/dedup, shape-mismatch, missing-required, unexpected-extra,
  tied-head-omitted, and tied-head-differs.
- **`run` end-to-end** on a synthetic model (no assets, fresh-checkout safe) proves
  both the clean exit and the shape-mismatch exit-1 path.
- **The reality anchor:** the real model cross-checks clean at **311 /
  751,632,384 / 596,049,920 / 26.1%** — every figure derived from `config.json`.

Reality checks skip gracefully when the (git-ignored) assets aren't fetched, so a
fresh checkout stays green.

---

## Cross-links

- 📖 *Inference Engineering* §4.2.2 "Model File Formats" (p.103) — safetensors/GGUF;
  §2.2.1 (p.49) — LLM architecture.
- 🔧 `ds4` — `ds4.c` owns GGUF loading (mmap-based); `gguf-tools/` is the parser we'll
  read at **M5** when quantization is the lesson. ds4 is GGUF-only; we chose
  safetensors for M1–M4 (Qwen ships it; clean bf16).
- 🧭 Raschka's [workflow](https://magazine.sebastianraschka.com/p/workflow-for-understanding-llms)
  — technical report → config → reference implementation.
- Learnings: [01 · formats](learnings/01-safetensors-vs-gguf.md) ·
  [05 · shapes](learnings/05-reading-shapes.md) · [06 · mmap](learnings/06-mmap.md) ·
  [07 · bf16](learnings/07-bf16.md) · [09 · config](learnings/09-config.md) ·
  [10 · block anatomy](learnings/10-transformer-block-anatomy.md).
- ➡ **M2 — forward pass → logits**: where these tensors finally *compute*.
