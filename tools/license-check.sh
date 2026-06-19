#!/usr/bin/env bash
set -euo pipefail

# Fail on GPL-family/copyleft dependencies. Warn on unknown or weak-copyleft
# license metadata so we can review manually before accepting a dependency.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'PY'
import json
import re
import subprocess
import sys

GPL_RE = re.compile(r"\b(?:AGPL|GPL|LGPL)\b|\b(?:AGPL|GPL|LGPL)-", re.I)
WEAK_RE = re.compile(r"\b(?:MPL|EPL|CDDL)\b|\b(?:MPL|EPL|CDDL)-", re.I)

errors: list[str] = []
warnings: list[str] = []

# Rust: Cargo metadata includes transitive crates and their declared license
# expressions when crates.io provides them.
cargo = subprocess.run(
    ["cargo", "metadata", "--format-version", "1"],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
)
for pkg in json.loads(cargo.stdout)["packages"]:
    name = pkg["name"]
    version = pkg["version"]
    license_text = (pkg.get("license") or "").strip()
    label = f"Rust {name} {version}"
    if not license_text:
        warnings.append(f"{label}: UNKNOWN license metadata")
    elif GPL_RE.search(license_text):
        errors.append(f"{label}: prohibited GPL-family license: {license_text}")
    elif WEAK_RE.search(license_text):
        warnings.append(f"{label}: weak-copyleft/manual-review license: {license_text}")

# Python: inspect installed distributions in the uv-managed scripts env. This
# does not use pip and does not alter pyproject.toml/uv.lock.
py_code = r'''
import json
from importlib.metadata import distributions
rows = []
for dist in distributions():
    md = dist.metadata
    rows.append({
        "name": md.get("Name", "<unknown>"),
        "version": dist.version,
        "license": md.get("License") or "",
        "classifiers": md.get_all("Classifier") or [],
    })
print(json.dumps(rows))
'''
uv = subprocess.run(
    ["uv", "run", "--directory", "scripts", "python", "-c", py_code],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
)
for pkg in json.loads(uv.stdout):
    name = pkg["name"]
    version = pkg["version"]
    license_bits = [pkg.get("license") or ""] + pkg.get("classifiers", [])
    license_text = " | ".join(bit.strip() for bit in license_bits if bit and bit.strip())
    label = f"Python {name} {version}"
    if not license_text:
        warnings.append(f"{label}: UNKNOWN license metadata")
    elif GPL_RE.search(license_text):
        errors.append(f"{label}: prohibited GPL-family license: {license_text}")
    elif WEAK_RE.search(license_text):
        warnings.append(f"{label}: weak-copyleft/manual-review license: {license_text}")

if warnings:
    print("license-check warnings:", file=sys.stderr)
    for warning in warnings:
        print(f"  WARN {warning}", file=sys.stderr)

if errors:
    print("license-check errors:", file=sys.stderr)
    for error in errors:
        print(f"  FAIL {error}", file=sys.stderr)
    sys.exit(1)

print("license-check: no GPL-family dependencies found")
PY
