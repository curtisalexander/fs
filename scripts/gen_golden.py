#!/usr/bin/env python3
"""Generate golden token-ID vectors from the OFFICIAL Qwen3 tokenizer.

These are the reference our hand-written Rust BPE must reproduce *exactly* — the
M0 "verify" step (round-trip + match official IDs). We load the real
`tokenizer.json` via Hugging Face's `tokenizers` library, encode a spread of
deliberately tricky strings, and write the results to
`tests/golden/tokenizer.json` (committed, so `cargo test` needs no Python).

    uv run --directory scripts fetch_model.py   # once, to get tokenizer.json
    uv run --directory scripts gen_golden.py

Idempotent: same tokenizer + same SAMPLES => byte-identical output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

REPO_ID = "Qwen/Qwen3-0.6B"
REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_JSON = REPO_ROOT / "models" / "qwen3-0.6b" / "tokenizer.json"
OUT = REPO_ROOT / "tests" / "golden" / "tokenizer.json"

# Deliberately tricky inputs. Byte-level BPE is sensitive to leading spaces,
# multibyte UTF-8, astral-plane emoji, digits, and newlines/tabs — so probe them.
SAMPLES: list[str] = [
    "hello world",
    " hello world",          # leading space attaches to the next token
    "The capital of France is",
    "fs: a failed star",
    "café résumé naïve",     # latin-1 accents (2-byte UTF-8)
    "日本語のテキスト",        # CJK (3-byte UTF-8)
    "emoji 🚀🌟 test",         # astral-plane (4-byte UTF-8)
    "def f(x):\n    return x * 2",  # code: newline + spaces
    "tab\tseparated",        # a literal tab
    "GPT-4 & Qwen3-0.6B",
    "    four leading spaces",
    "trailing space ",
    "Numbers 1234567890",
    "Mixed: ASCII, café, 日本, 🚀",
]


def main() -> int:
    if not TOKENIZER_JSON.exists():
        print(
            f"missing {TOKENIZER_JSON}\n"
            f"  run first:  uv run --directory scripts fetch_model.py",
            file=sys.stderr,
        )
        return 1

    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    cases = []
    for text in SAMPLES:
        enc = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(enc.ids, skip_special_tokens=False)
        cases.append({"text": text, "ids": enc.ids, "decoded": decoded})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": REPO_ID,
        "source": "tokenizer.json",
        "add_special_tokens": False,
        "cases": cases,
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases -> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
