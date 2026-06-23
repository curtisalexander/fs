#!/usr/bin/env python3
"""Idempotent fetch of Qwen3-0.6B assets from the Hugging Face Hub.

By default this pulls only the small tokenizer/config files M0 needs (~16 MB).
Pass --weights to also pull model.safetensors (~1.5 GB), needed from M1 on.

Run via uv (deps are pinned in pyproject.toml / uv.lock):

    uv run --directory scripts fetch_model.py            # tokenizer assets only
    uv run --directory scripts fetch_model.py --weights  # + 1.5 GB weights

Re-running is a no-op: huggingface_hub skips files already present and verified.
Everything lands in  models/qwen3-0.6b/  (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen3-0.6B"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "models" / "qwen3-0.6b"

# Small text/JSON the tokenizer (M0) needs, plus configs we reuse from M1 on.
# We parse tokenizer.json directly (it carries vocab + merges + the pre-tokenizer
# regex + special tokens), so the separate GPT-2 vocab.json/merges.txt are no
# longer fetched — tokenizer.json supersedes them.
TOKENIZER_FILES = [
    "tokenizer.json",          # all-in-one HF fast tokenizer (vocab+merges+rules+specials)
    "tokenizer_config.json",   # special-token wiring + chat template (used from M3)
    "config.json",             # vocab_size (151936) and architecture dims
    "generation_config.json",  # sampling defaults (used at M3)
]

# Large weights — deferred until M1 (loading / forward pass).
WEIGHT_FILES = ["model.safetensors"]


def fetch(files: list[str]) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in files:
        print(f"  - {name} ... ", end="", flush=True)
        path = hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=DEST)
        size_mb = Path(path).stat().st_size / 1e6
        print(f"ok ({size_mb:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--weights",
        action="store_true",
        help="also download model.safetensors (~1.5 GB)",
    )
    args = ap.parse_args()

    files = list(TOKENIZER_FILES)
    if args.weights:
        files += WEIGHT_FILES

    print(f"Fetching {REPO_ID} -> {DEST.relative_to(REPO_ROOT)}/")
    fetch(files)
    print("Done. (re-run anytime — already-present files are skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
