# RESOURCES — the cross-reference index

The three sources Failed Star is built from, indexed so any milestone doc can link
precisely. When you write a doc, cite **book §+page**, **`ds4` file**, and
**Raschka** where each applies.

---

## 1. The book — *Inference Engineering*, Philip Kiely (Baseten, 2026)

Local file: [`../Inference Engineering.pdf`](../Inference%20Engineering.pdf) (259 pages).
A production-serving survey: it teaches the concepts and vocabulary; it treats the
engine mostly as a box you operate. Most relevant chapters for us: **2, 4, 5**.

Get your own copy: peruse the free [interactive guide](https://inferenceengineering.tech/)
(animated diagrams + VRAM / arithmetic-intensity / KV-cache calculators), then buy
the paperback from [Baseten Books](https://www.baseten.co/inference-engineering/).

> Reading PDFs in this repo: the built-in reader needs `poppler` (`brew install
> poppler`); extract a page range with `pdftotext -f <first> -l <last> "Inference
> Engineering.pdf" -`.

### Page index (the parts we actually use)
| § | Topic | Page | Milestone |
|---|---|---:|---|
| Ch 0 | Inference (what it is) | 15 | — |
| Ch 1 | Prerequisites | 23 | — |
| 1.4 | Measuring Latency & Throughput | 35 | M4, M6 |
| Ch 2 | **Models** | 39 | M2 |
| 2.1 | Neural Networks | 42 | M2 |
| 2.1.1 | Linear Layers and Matmul | 44 | M2 |
| 2.1.2 | Activation Functions | 44 | M2 |
| 2.2 | **LLM Inference Mechanics** | 46 | M0, M2, M3 |
| 2.2.1 | LLM Architecture | 49 | M1, M2 |
| 2.2.2 | Transformer Blocks | 50 | M2 |
| 2.2.3 | **Attention** | 52 | M2 |
| 2.2.4 | Mixture of Experts | 53 | M7 |
| 2.4 | **Calculating Inference Bottlenecks** | 61 | (read for M4/M6) |
| 2.4.1 | Ops:Byte Ratio & Arithmetic Intensity | 62 | M4, M6 |
| 2.4.2 | LLM Inference Bottlenecks | 63 | M4, M6 |
| 2.5 | Optimizing Attention | 67 | M2, M6 |
| Ch 3 | **Hardware** | 71 | M6 |
| 3.1 | GPU Architecture (compute 74 / memory 76) | 74 | M6 |
| 3.5 | **Local Inference** (desktop 90 / mobile 91) | 89 | M6 |
| Ch 4 | **Software** | 93 | M6 |
| 4.1 | CUDA (kernels 98 / **fusion 100**) | 96 | M6 |
| 4.2.2 | Model File Formats | 103 | M1 |
| 4.3 | Inference Engines (vLLM 106 …) | 105 | (context) |
| Ch 5 | **Techniques** | 117 | M4, M5, M7 |
| 5.1 | **Quantization** (formats 121 / approaches 125 / quality 128) | 120 | M5 |
| 5.2 | Speculative Decoding | 129 | M7 |
| 5.3 | **Caching** (reuse 136 / where 139 / long-ctx 141) | 136 | M4 |
| 5.4 | Model Parallelism | 142 | (context) |
| Ch 6 | Modalities | 153 | — |
| Ch 7 | Production | 177 | — |
| App. A | Glossary | 209 | reference |
| App. B | Recommended Reading | 231 | reference |

---

## 2. `ds4` — Dwarf Star, antirez · `reference/ds4/`

Repo: <https://github.com/antirez/ds4>. Self-contained C engine for
**DeepSeek-V4-Flash** (284B total / 13B active MoE, 1M ctx, compressed attention).
**Metal is the primary backend.** Our reference for "how a pro does it." We *read*
it; we don't build its CUDA/ROCm/server/agent/distributed parts.

### File map (what to open for what)
| Area | Files | Notes |
|---|---|---|
| Core engine | `ds4.c` (~26k LOC), `ds4.h` | orchestration, tokenizer, graph |
| **Metal backend** | `ds4_metal.m` (~26k LOC), `ds4_gpu.h` | host-side Metal via ObjC |
| **Metal kernels** | `metal/` (19 `.metal` files) | one op each — see below |
| Tokenizer | `ds4.c` | byte-level BPE; `str_i32_table` hash table for token→id |
| Agent memory store | `rax.c/.h` | radix tree (antirez's `rax`), used from `ds4_server.c` |
| KV cache | `ds4_kvstore.c/.h` | "KV cache as a disk citizen" |
| SSD streaming | `ds4_ssd.c/.h` | stream weights/KV from SSD |
| Quantization | `gguf-tools/` (+ `imatrix/`, `quality-testing/`) | GGUF gen + calibration |
| CLI / REPL | `ds4_cli.c`, `linenoise.c/.h` | line editing |
| Server | `ds4_server.c`, `ds4_web.c/.h` | OpenAI/Anthropic-compat API |
| Agent | `ds4_agent.c`, `AGENT.md` | built-in coding agent |
| Eval / bench | `ds4_eval.c`, `ds4_bench.c`, `speed-bench/` | validation + perf |
| Distributed | `ds4_distributed.c/.h` | multi-machine TCP |
| Other backends | `ds4_cuda.cu`, `ds4_rocm.cu`, `rocm/` | out of scope for us |
| Model notes | `MODEL_CARD.md`, `README.md`, `STRIXHALO.md` | great reading |
| Steering | `dir-steering/` | activation steering vectors |

### Metal kernel map (`reference/ds4/metal/`) — our M6 rosetta stone
| Shader | Op | Our milestone |
|---|---|---|
| `get_rows.metal` | embedding/row gather | M2 |
| `norm.metal` | RMSNorm | M2 |
| `dsv4_rope.metal` | RoPE | M2 |
| `dense.metal` | dense matmul | M2 |
| `flash_attn.metal` | attention (flash) | M2/M6 |
| `softmax.metal` | softmax | M2/M3 |
| `glu.metal` | SwiGLU/GLU FFN | M2 |
| `moe.metal` | MoE routing | M7 |
| `argsort.metal` | sort (top-k sampling) | M3 |
| `dsv4_kv.metal` | compressed KV | M4/M7 |
| `dsv4_hc.metal`, `dsv4_misc.metal` | DeepSeek compressed attn | M7 |
| `cpy / concat / repeat / set_rows / sum_rows / unary / bin` | tensor plumbing | as needed |

---

## 3. Sebastian Raschka — architecture context

- **The Big LLM Architecture Comparison** —
  <https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison>
  Tour of modern design choices: MHA/GQA/**MLA**, sliding-window/sparse/linear
  attention, pre/post-norm, QK-norm, RoPE/**NoPE**/YaRN, SwiGLU, **MoE**
  (shared-expert / few-large vs many-small). Covers DeepSeek V3 (MLA+MoE), Qwen3,
  Llama 4, Gemma 3, GPT-OSS, Kimi, etc. **Use to place our model's choices in
  context, and to understand `ds4`'s DeepSeek-V4 (MLA-family) attention.**
- **LLM Architecture Gallery** — <https://sebastianraschka.com/llm-architecture-gallery/>
  Visual reference; good source for diagrams to adapt in our HTML docs.
- **Workflow for Understanding LLMs** —
  <https://magazine.sebastianraschka.com/p/workflow-for-understanding-llms>
  His method: technical report → HF config files → reference implementation
  ("working code doesn't lie") → implement a few by hand. We adopt this per
  milestone (see `docs/00-map.md` §6).
- *(optional, paid — not a dependency)* His book **"Build a Large Language Model
  (From Scratch)"** — a clean book-length from-scratch reference (BPE, attention,
  training). It covers the same ground as Karpathy's free Zero-to-Hero + `llama2.c`;
  the three free articles above are what we actually cross-reference. Reach for the
  book only if you want the long-form, sit-down treatment.

---

## 4. Model / tooling references

### Qwen3-0.6B — our starter model (the provenance chain)

The architecture we implement was **designed and trained by the Qwen team
(Alibaba)** and *documented*, then implemented in the reference library. We can't
derive it from the config (a config only *parameterizes* an architecture whose
structure lives in code — see [`learnings/09-config.md`](learnings/09-config.md));
we read the paper + reference code and verify by reproducing numbers. The full
"who decided this and how do we know" story is
[`learnings/10-transformer-block-anatomy.md`](learnings/10-transformer-block-anatomy.md).

| What | Link | Role for us |
|---|---|---|
| Model card + assets | <https://huggingface.co/Qwen/Qwen3-0.6B> | the `config.json`, `tokenizer.json`, `model.safetensors` we load |
| **Technical report** | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) | the *design*: why QK-norm, GQA, SwiGLU — the creators' word |
| **Reference code** (generated) | [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | the **spec** we read + the **oracle** we'll run at M2 |
| Reference code (true source) | [`modular_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modular_qwen3.py) | `modeling_*.py` is auto-generated from this |
| Config class | [`configuration_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/configuration_qwen3.py) | every field of `config.json`, with defaults |

> The reference is copyrighted **"2025 The Qwen team, Alibaba Group and the
> HuggingFace Inc. team"** — the chain of trust (creator → reference lib → us) is
> right there in the file header. We **read** it as spec and **run** it as oracle;
> we never port its code. We copy the *architecture* (the weights demand it), never
> the *code* (we prove equivalence with golden numbers). See learning 10.

### Other tooling (fill in as we reach it)
- Rust ↔ Metal via ObjC runtime FFI (`objc2` runtime; we avoid the `metal`
  convenience crate) — *capture the exact approach at M6.*
- Golden-vector script (HF `transformers` one-shot logit dump) — *add at M2.*
