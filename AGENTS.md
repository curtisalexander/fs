# AGENTS.md — operating contract for Failed Star (`fs`)

Tight on purpose. This file is auto-loaded every session (Claude Code reads it via
the `CLAUDE.md → AGENTS.md` symlink). It points at the detail; it does not inline it.

## Start here, every session

1. Read [`PROGRESS.md`](PROGRESS.md) — where we are, last session, open decisions, next step.
2. Re-anchor with [`PLAN.md`](PLAN.md) — the M0→M7 milestone curriculum.
3. Follow [`docs/dev-loop.md`](docs/dev-loop.md) — the full session ritual, local
   checks, Python oracle, dependency/license policy, and the docs/site sync loop.

## Core invariants (the "spirit" — keep these honest)

- **Discuss before committing.** Teach-as-we-build. Tight, readable, no hidden
  abstraction (the `ds4` ethos).
- **Learning-first, then fast.** Correctness and clarity come first; making it go
  fast is a separate, later lesson — never trade away clarity for speed in the
  first pass.
- **Correctness = golden vectors** from the official implementation. Python is only
  ever a one-shot oracle (golden data / asset fetch), never a second engine.
- **Shapes are always explicit.** Anytime tensors/matrices/vectors appear — in
  docs, CLI output, or code — make the dimensions clear and easy to *see* (legends,
  mini diagrams, `in→out` views) and back them with asserts that fail loudly.
- **Scope = Metal/macOS only.** Rust host + MSL kernels via raw FFI (no wrapper).

## Leaving the repo

- Update [`PROGRESS.md`](PROGRESS.md) (what changed, decisions, next smallest step).
- Keep `cargo build` green unless `PROGRESS.md` says otherwise.
- Markdown is the source of truth; the site is hand-distilled HTML. New/updated
  **learnings graduate into the site's Learnings section** and get linked from the
  doc that references them (see `docs/dev-loop.md` → Docs/site loop).
