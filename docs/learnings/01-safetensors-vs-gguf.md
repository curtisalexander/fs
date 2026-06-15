# Learning 01 — Model file formats: safetensors vs GGUF

> **Date:** 2026-06-13 · **Context:** deciding M1's on-disk format · **Status:** decided
>
> 📖 *Inference Engineering* §4.2.2 "Model File Formats" (p.103)
> 🔧 `ds4`: `ds4.c` ("owns GGUF loading"), `gguf-tools/`
> 🧭 Raschka: ["Workflow for Understanding LLMs"](https://magazine.sebastianraschka.com/p/workflow-for-understanding-llms) — *inspect the config files / "working code doesn't lie"*

This is the first thing we learned: **where do a model's weights actually live on
disk, and in what shape?** A model is, physically, two things — a pile of numbers
(the weights) and a description of how they're arranged (names, shapes, dtypes).
A *file format* is just an agreed way to write those two things down. Two formats
dominate, built for different jobs.

---

## safetensors — HuggingFace's native format

This is what **Qwen3-0.6B ships as** on Hugging Face. It is almost shockingly
simple — which is exactly why it's a great first format to parse by hand:

```
┌──────────────┬───────────────────────────┬───────────────────────────┐
│ 8 bytes      │ N bytes                   │ the rest of the file      │
│ u64 (LE) = N │ JSON header               │ raw tensor bytes          │
└──────────────┴───────────────────────────┴───────────────────────────┘
```

1. **First 8 bytes:** a little-endian `u64` giving the length `N` of the header.
2. **Next `N` bytes:** a UTF-8 **JSON object**. Each key is a tensor name; each
   value is `{ "dtype": ..., "shape": [...], "data_offsets": [start, end] }`.
   (An optional `__metadata__` key holds arbitrary string key/values.)
3. **Everything after:** one contiguous blob of raw tensor data. Each tensor lives
   at `[start, end)` *within that blob*, row-major (C-contiguous).

That's the entire format. Properties that matter to us:

- **Weights are in their original dtype** — Qwen3 is **bf16**. No quantization,
  no decoding scheme: the bytes *are* the numbers. (We'll convert bf16→f32 when we
  compute; that's trivial.)
- **It is "safe":** unlike Python `pickle`/`.bin` checkpoints, there's no embedded
  code to execute — just data. Hence the name.
- **Zero-copy / mmap-friendly:** you can map the file and point your tensors
  straight at the bytes. (`ds4` mmaps too; the idea is the same.)
- **It does NOT contain the tokenizer or hyperparameters.** Those live in sibling
  files: `config.json` (layers, dims, heads, vocab size, …) and `tokenizer.json`
  (the vocab + merges we need for M0). We need those files anyway.

## GGUF — llama.cpp's format (what `ds4` uses)

GGUF is built for *local inference distribution*: get a model running from **one
self-contained file**. It carries everything.

```
┌─ header ───────────────────────────────────────────────────────────────┐
│ magic "GGUF" (0x46554747)  · version (u32) · n_tensors · n_metadata_kv │
├─ metadata (key/value pairs) ───────────────────────────────────────────┤
│ general.architecture, hyperparameters, AND the full tokenizer/vocab,   │
│ chat template, etc. (~13 typed value kinds, arrays supported)          │
├─ tensor info ──────────────────────────────────────────────────────────┤
│ per tensor: name · n_dims · dims[] · ggml_type (quant kind) · offset   │
├─ padding to alignment ─────────────────────────────────────────────────┤
│ tensor data: quantized blocks (Q4_K, Q2_K, IQ2_XXS, …) or f16/f32      │
└────────────────────────────────────────────────────────────────────────┘
```

Key differences from safetensors:

- **One file holds it all** — weights *and* tokenizer *and* config *and* chat
  template. Download one `.gguf`, run.
- **Quantization is native.** Tensors aren't plain arrays; they're stored in
  **block quant formats** — a block of N weights shares a scale (and sometimes a
  min/secondary scale), packed into far fewer bits. `Q4_K` ≈ 4 bits/weight,
  `Q2_K`/`IQ2_XXS` ≈ 2 bits/weight. This is *the* reason a huge model fits a
  laptop — and it's a whole topic of its own (our M5).
- **More complex to parse:** a binary key/value metadata section with many value
  types, plus the per-format block layouts you must decode to get real numbers.

### Evidence: `ds4` is GGUF, end to end

- `ds4.c:5` — *"This file is deliberately vertical: it owns GGUF loading…"*
- `ds4.c:11` — *"Loading is mmap based. The loader parses only the GGUF header,
  metadata…"*
- `ds4.c:579` — `#define DS4_GGUF_MAGIC 0x46554747u  /* "GGUF", little endian. */`
- `ds4.c:1514` — the `GGUF_VALUE_*` type enum (UINT8…FLOAT32…).
- `download_model.sh` pulls only `*.gguf` files; `gguf-tools/` builds/quantizes GGUF.

So `ds4` has **no safetensors path at all** — it's a GGUF engine.

---

## The tradeoff, for *our* goals

| | safetensors | GGUF |
|---|---|---|
| Qwen3-0.6B ships it? | ✅ natively | ❌ needs conversion (llama.cpp) |
| Parse-from-scratch effort | trivial (len + JSON + bytes) | meatier (binary KV + quant blocks) |
| Weights for a first forward pass | clean bf16, no decoding | fine if f16, but quant blocks loom |
| Includes tokenizer/config? | ❌ (sibling files) | ✅ (all in one) |
| Quantization | none (add later) | native, the whole point |
| Matches `ds4` directly | ❌ | ✅ |

---

## Our decision (and why)

**safetensors for M1–M4, then GGUF at M5.**

The reasoning is **sequencing complexity to the milestone that needs it**:

- **M1 should teach "weights are just bytes + a shape table."** safetensors makes
  that idea visceral in an afternoon, and Qwen ships it natively, so there's *no
  tooling detour* before we've loaded a single tensor.
- **M2–M4 stay clean.** Real bf16 weights mean our first-ever forward pass isn't
  also fighting dequantization math while we hunt numerical bugs against the golden
  vector.
- **M5 is where quantization becomes the lesson** — and GGUF *is* a quantization
  format. Writing the GGUF loader *there*, side by side with `ds4`'s parser and
  `gguf-tools/`, is the richest version of that comparison. ➡️ **Two formats =
  two distinct lessons** ("load raw tensors" vs "load quantized tensors"), not
  wasted work.

We stay honest with `ds4` the whole time: it's GGUF-only; we converge with it at M5.

---

## Mental model to keep

> A model on disk = **(numbers) + (a table describing the numbers)**.
> - **safetensors** writes the numbers *plainly*. Simple; the bytes are the values.
> - **GGUF** writes the numbers *compressed* (quantized) and bundles the tokenizer
>   and config alongside. Self-contained; the bytes need *decoding* to become values.

The leap from the first to the second — "the bytes need decoding" — is the door
into quantization, which is most of what makes local inference *possible*. We walk
through that door deliberately at M5.

---

### To revisit at M5
- Decode at least `Q4_K` and a 2-bit format (`Q2_K` / `IQ2_XXS`) by hand; compare
  block layouts against `gguf-tools/quants.c` and `ds4.c`'s block format section
  (around `ds4.c:312`).
- Measure: memory + decode tok/s + quality (perplexity) vs our bf16 baseline.
- 📖 then-relevant reading: §5.1 "Quantization" (p.120).
