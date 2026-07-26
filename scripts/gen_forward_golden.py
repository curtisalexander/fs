#!/usr/bin/env python3
"""Generate M2 forward-pass checkpoints from the official Qwen3 implementation.

The Rust engine uses these as a layered correctness oracle: if final logits do
not match, embedding / block-0 / final-norm checkpoints identify the first stage
that diverged. Hugging Face runs once, here; it is never a second engine.

Prerequisites and use:

    uv run --directory scripts --frozen fetch_model.py --weights
    uv run --directory scripts --frozen gen_forward_golden.py

Outputs are raw little-endian f32 files plus a JSON manifest in
``tests/golden/forward/``. Re-running on the same recorded runtime/platform with
the same assets and locked dependencies is byte-identical; committed checksums
are canonical when another CPU platform differs within the documented tolerance.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROMPT = "The capital of France is"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "qwen3-0.6b"
OUT_DIR = REPO_ROOT / "tests" / "golden" / "forward"


def sha256(path: Path) -> str:
    """Hash a source asset so the manifest identifies the exact model bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_dtypes(path: Path) -> set[str]:
    """Read safetensors' JSON header and return all source tensor dtypes."""
    with path.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_len))
    return {entry["dtype"] for name, entry in header.items() if name != "__metadata__"}


def tensor_output(output: Any) -> torch.Tensor:
    """Unwrap a hook output while remaining clear about the tensor boundary."""
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"hook produced {type(value).__name__}, expected Tensor")
    return value


def save_f32(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Write one checkpoint and return its shape/checksum manifest entry."""
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    array = value.numpy().astype("<f4", copy=False)
    data = array.tobytes(order="C")
    path = OUT_DIR / f"{name}.f32"
    path.write_bytes(data)
    return {
        "file": path.name,
        "dtype": "f32-le",
        "shape": list(value.shape),
        "elements": value.numel(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    required = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "tokenizer.json",
        MODEL_DIR / "tokenizer_config.json",
        MODEL_DIR / "model.safetensors",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("missing model assets:", file=sys.stderr)
        for path in missing:
            print(f"  - {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(
            "run first: uv run --directory scripts --frozen fetch_model.py --weights",
            file=sys.stderr,
        )
        return 1

    # CPU + fp32 + eager attention is intentionally slow and plain: no mixed
    # precision, device-specific fused kernel, cache mutation, or training mode
    # can move the reference numbers underneath us.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    model.to("cpu")
    model.eval()

    load_problems = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if load_problems:
        raise ValueError(f"model did not load exactly: {load_problems}")
    if checkpoint_dtypes(MODEL_DIR / "model.safetensors") != {"BF16"}:
        raise ValueError("expected every source weight tensor to be BF16")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("expected every materialized model parameter to be CPU float32")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("expected lm_head and embed_tokens to be tied in memory")

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    encoded = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded["attention_mask"].to("cpu")
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    cache_position = position_ids.squeeze(0)
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            captured[name] = tensor_output(output).detach().to("cpu", torch.float32).clone()

        return hook

    handles = [
        model.model.embed_tokens.register_forward_hook(capture("embedding")),
        model.model.layers[0].register_forward_hook(capture("block_0")),
        model.model.norm.register_forward_hook(capture("final_norm")),
    ]
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
                logits_to_keep=1,
            )
            captured["logits"] = output.logits[:, -1, :].detach().to("cpu", torch.float32).clone()
    finally:
        for handle in handles:
            handle.remove()

    batch, seq = input_ids.shape
    hidden = model.config.hidden_size
    vocab = model.config.vocab_size
    expected_shapes = {
        "embedding": (batch, seq, hidden),
        "block_0": (batch, seq, hidden),
        "final_norm": (batch, seq, hidden),
        "logits": (batch, vocab),
    }
    for name, expected in expected_shapes.items():
        got = tuple(captured[name].shape)
        if got != expected:
            raise ValueError(f"{name}: expected shape {expected}, got {got}")

    # Rust has no batch axis in M2, so strip the known-singleton batch dimension
    # before writing: [1,seq,H] -> [seq,H], [1,V] -> [V].
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        name: save_f32(name, captured[name].squeeze(0))
        for name in ("embedding", "block_0", "final_norm", "logits")
    }
    manifest = {
        "schema": 1,
        "model": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "config_sha256": sha256(MODEL_DIR / "config.json"),
            "tokenizer_sha256": sha256(MODEL_DIR / "tokenizer.json"),
            "tokenizer_config_sha256": sha256(MODEL_DIR / "tokenizer_config.json"),
            "weights_sha256": sha256(MODEL_DIR / "model.safetensors"),
        },
        "input": {
            "prompt": PROMPT,
            "add_special_tokens": False,
            "input_ids": input_ids.squeeze(0).tolist(),
            "attention_mask": attention_mask.squeeze(0).tolist(),
            "position_ids": position_ids.squeeze(0).tolist(),
            "cache_position": cache_position.tolist(),
        },
        "reference": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "os": platform.system(),
            "architecture": platform.machine(),
            "device": "cpu",
            "checkpoint_dtype": "bfloat16",
            "parameter_dtype": "float32",
            "activation_dtype": "float32",
            "attention_implementation": "eager",
            "eval": True,
            "inference_mode": True,
            "use_cache": False,
            "logits_to_keep": 1,
            "deterministic_algorithms": True,
            "torch_threads": torch.get_num_threads(),
            "lm_head": "tied to model.embed_tokens.weight",
        },
        "fixture": {
            "layout": "C-row-major",
            "batch_axis_removed": True,
            "byte_order": "little-endian",
        },
        "tolerance": {"absolute": 1e-4, "relative": 1e-4},
        "checkpoints": checkpoints,
    }
    boundaries = {
        "embedding": (["sequence", "hidden"], "embedding output before decoder layer 0"),
        "block_0": (["sequence", "hidden"], "decoder layer 0 output after MLP residual"),
        "final_norm": (["sequence", "hidden"], "final RMSNorm output"),
        "logits": (["vocabulary"], "lm_head output for the last sequence position"),
    }
    for name, (axes, boundary) in boundaries.items():
        manifest["checkpoints"][name]["axes"] = axes
        manifest["checkpoints"][name]["boundary"] = boundary
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(entry["bytes"] for entry in checkpoints.values())
    print(f"prompt: {PROMPT!r}")
    print(f"tokens: {input_ids.squeeze(0).tolist()}")
    for name, entry in checkpoints.items():
        print(f"  {name:10} {entry['shape']} -> {entry['file']} ({entry['bytes']:,} bytes)")
    print(f"wrote {total_bytes:,} checkpoint bytes + manifest -> {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
