# Dev loop

This repo is a slow, multi-session learning project. When coming back after a
break, use this page to reload the context and run the small checks before
changing code.

## Start of session

1. Read [`PROGRESS.md`](../PROGRESS.md) for the current milestone, last session,
   open decisions, and next steps.
2. Skim [`PLAN.md`](../PLAN.md) only if you need to re-anchor the milestone map.
3. If working on a concept-heavy piece, re-open the relevant learning note in
   [`docs/learnings/`](learnings/) and the relevant reference files in
   [`reference/ds4/`](../reference/ds4/).

## Local checks

```sh
rustc --version
cargo --version
uv --version
cargo build
cargo test
```

The Rust edition is set in [`Cargo.toml`](../Cargo.toml). This project uses
edition **2024**, currently the newest Rust edition. During early M0,
`cargo build` is expected to pass while unfinished methods may still be
`todo!()`. Once a helper is implemented, add or run the matching tests before
moving to the next helper.

## Python oracle / data pipeline

Python is used only to fetch model assets and generate golden reference data.
Python dependency management **must use uv and `pyproject.toml`**. Do not use
`pip`, `requirements.txt`, ad-hoc virtualenv commands, or inline script metadata
for this repo. The environment is pinned by
[`scripts/pyproject.toml`](../scripts/pyproject.toml) and
[`scripts/uv.lock`](../scripts/uv.lock).

```sh
# Fetch tokenizer/config assets into models/qwen3-0.6b/ (ignored by git)
uv run --directory scripts fetch_model.py

# Later, fetch model weights too
uv run --directory scripts fetch_model.py --weights

# Regenerate committed tokenizer golden vectors
uv run --directory scripts gen_golden.py
```

The generated oracle fixtures that tests rely on live under
[`tests/golden/`](../tests/golden/). Scratch or bulky generated files should not
be committed.

## Dependency freshness and age gate

We want dependencies to stay current, but not adopt packages immediately after
publication. Use a **7-day age gate** for routine updates.

### Rust

Check the toolchain and edition:

```sh
rustup update stable
rustc --version
cargo --version
rg '^edition' Cargo.toml
```

Check what Cargo would update within the existing semver constraints:

```sh
cargo update --dry-run
```

Cargo does not currently have a built-in `--exclude-newer` age gate like uv. For
Rust dependency updates, the safe manual loop is:

1. run `cargo update --dry-run`,
2. inspect the proposed package versions,
3. check publish dates on crates.io,
4. only then run `cargo update` or `cargo update -p <crate>`.

For a stricter workflow, add a small helper later that queries the crates.io API
for the dry-run versions and refuses any version published less than 7 days ago.

### Python

Check what is outdated:

```sh
uv tree --directory scripts --outdated
```

uv supports the 7-day age gate directly with `--exclude-newer`. Always run the
`--dry-run` first. If the lockfile already contains a version newer than the
cutoff, uv may propose a downgrade; review that intentionally rather than
blindly applying it.

```sh
# macOS/BSD date
cutoff=$(date -u -v-7d '+%Y-%m-%dT%H:%M:%SZ')
uv lock --directory scripts --upgrade --exclude-newer "$cutoff" --dry-run
uv lock --directory scripts --upgrade --exclude-newer "$cutoff"

# Linux/GNU date equivalent
cutoff=$(date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%SZ')
uv lock --directory scripts --upgrade --exclude-newer "$cutoff" --dry-run
uv lock --directory scripts --upgrade --exclude-newer "$cutoff"
```

Then run:

```sh
uv run --directory scripts python --version
tools/license-check.sh
cargo test
```

## License policy

This repo is MIT licensed. Do **not** add GPL-family/copyleft dependencies that
would complicate or pollute the license story.

Policy:

- Allowed by default: permissive licenses such as MIT, Apache-2.0, BSD, ISC,
  Zlib, and Unlicense.
- Prohibited by default: GPL, AGPL, and LGPL dependencies, direct or transitive.
- Manual review required: weak/file-level copyleft or unusual licenses such as
  MPL, EPL, CDDL, custom license text, or missing/unknown metadata.
- If a dependency is kept after manual review, document why in `PROGRESS.md` or
  the relevant milestone doc.

Run the license check after adding or updating dependencies:

```sh
tools/license-check.sh
```

The check inspects transitive Rust crates via `cargo metadata` and Python
packages via the uv-managed `scripts` environment. It fails on GPL-family
licenses and warns on weak-copyleft or unknown metadata. Warnings are a prompt
for human review, not automatic approval.

## Docs/site loop

Working Markdown lives in `docs/*.md`; the public site is hand-authored HTML in
`docs/*.html` plus `docs/assets/`. HTML is a distillation, not an automatic build.
Check whether the site has drifted from its Markdown sources with:

```sh
tools/sync-check.sh
```

After deliberately re-distilling a page, stamp the ledger:

```sh
tools/sync-check.sh --update
```

## Dependency policy

The project prefers small, understandable code and avoids dependencies that hide
core inference concepts. But “zero dependencies” is not a religion.

A dependency is acceptable when it:

- handles a non-core side problem,
- improves correctness or portability,
- avoids a distracting implementation side quest,
- and is documented where the decision is made.

For example, hand-writing BPE is core to M0; hand-writing a full Unicode regex
engine is not, so M0 uses `fancy-regex` for Qwen's exact pre-tokenization
pattern. Small side quests are welcome when they are interesting, reasonable to
implement, and do not add much code or complexity.

## End of session

Before stopping:

1. Update [`PROGRESS.md`](../PROGRESS.md) with what changed, decisions made, and
   the next smallest step.
2. If a milestone doc changed, consider whether the HTML distillation needs a
   future sync note.
3. Leave the repo in a state where `cargo build` passes unless `PROGRESS.md`
   explicitly says otherwise.

### When a milestone flips (done / new "current")

Milestone status is shown in four hand-maintained places — bump all of them in
the same commit so they don't drift:

1. [`PLAN.md`](../PLAN.md) — the `☐ ◐ ☑` legend on the milestone heading.
2. [`PROGRESS.md`](../PROGRESS.md) — the **Current milestone** line at the top.
3. [`README.md`](../README.md) — the **Status** blurb and the milestone checklist.
4. [`docs/index.html`](index.html) — the **Build progress** strip: flip the
   `ms--done`/`ms--now` classes, the `progress-bar` width (done ÷ 7 core), and
   the caption.
