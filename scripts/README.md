# `scripts/` — setup & data generation (Python, via uv)

Everything here is **idempotent** (re-run safely) and **scriptable** (no manual
clicking). Python deps are pinned in [`pyproject.toml`](pyproject.toml) +
`uv.lock`; [`uv`](https://docs.astral.sh/uv/) builds the environment on demand,
so the only prerequisite is `uv` itself.

## Commands

```sh
# 1. Fetch Qwen3-0.6B tokenizer assets (~16 MB) into ../models/qwen3-0.6b/
uv run --directory scripts fetch_model.py

# 1b. (later, M1) also fetch the 1.5 GB weights
uv run --directory scripts fetch_model.py --weights

# 2. Generate golden token-ID vectors from the OFFICIAL tokenizer
#    -> ../tests/golden/tokenizer.json  (our M0 correctness oracle)
uv run --directory scripts gen_golden.py
```

`uv run` auto-syncs the locked environment first, so the first call may download
a managed Python + the deps; subsequent calls are instant.

## What's here

| File | Does |
|---|---|
| `fetch_model.py` | Downloads tokenizer/config files (and optionally weights) from the HF Hub. Skips files already present. |
| `gen_golden.py`  | Runs the official `tokenizers` library to produce reference encode/decode results our Rust impl must match. |
| `pyproject.toml` | Pinned dependency set (`huggingface_hub`, `tokenizers`). |

## Why Python here at all

Python is **only ever a one-shot oracle** — it produces reference data
(`tests/golden/…`) that our Rust engine is checked against. It is never a second
inference engine. See `PLAN.md` → "Decisions locked".
