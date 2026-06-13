# RESOURCES — the cross-reference index

The three sources Failed Star is built from, indexed so any milestone doc can link
precisely. When you write a doc, cite **book §+page**, **`ds4` file**, and
**Raschka** where each applies.

---

## 1. The book — *Inference Engineering*, Philip Kiely (Baseten, 2026)

Local file: [`../Inference Engineering.pdf`](../Inference%20Engineering.pdf) (259 pages).
A production-serving survey: it teaches the concepts and vocabulary; it treats the
engine mostly as a box you operate. Most relevant chapters for us: **2, 4, 5**.

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
| Tokenizer support | `rax.c/.h` | radix tree (antirez's `rax`) |
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
- His book **"Build a Large Language Model (From Scratch)"** — the canonical clean
  from-scratch reference (BPE, attention, training); great for M0/M2.

---

## 4. Model / tooling references (to fill in at M0/M1)
- Starter model card + `config.json` + tokenizer files (Llama-3.2-1B or
  Qwen3-0.6B) — *link once chosen.*
- Rust ↔ Metal via ObjC runtime FFI (`objc2` runtime; we avoid the `metal`
  convenience crate) — *capture the exact approach at M6.*
- Golden-vector script (HF `transformers` one-shot logit dump) — *add at M2.*
