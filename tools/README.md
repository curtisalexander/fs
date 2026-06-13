# `tools/` — site & sync helpers

The published site lives in [`../docs/`](../docs/) and is served by **GitHub Pages
straight from that folder** (Settings → Pages → *Deploy from a branch* → `main`
`/docs`). The empty `docs/.nojekyll` tells GitHub to serve our hand-written HTML
verbatim instead of running its own Jekyll build over it. No CI, no build step —
edit HTML/CSS, commit, push, it's live at
<https://curtisalexander.github.io/fs/>.

## The two layers (and how they stay "kinda sorta in sync")

| Layer | Where | Role |
|---|---|---|
| **Working copy** | `docs/*.md`, `docs/learnings/` | source of truth; messy, complete, evolving |
| **Distillation** | `docs/index.html`, `docs/assets/` | polished, curated, interactive — what readers see |

The HTML is **not** auto-generated from the markdown (no pandoc, no SSG). It's a
deliberate distillation: it can lag, summarize, re-order, and add interactive
diagrams the markdown can't. To keep it from silently going stale we use **drift
detection**, not conversion:

- [`sync-ledger.tsv`](sync-ledger.tsv) records, per HTML page, which markdown it
  distills and the commit at which they were last reconciled.
- [`sync-check.sh`](sync-check.sh) reports when a source `.md` has new commits
  since — i.e. when a page is due for a re-read.

```sh
tools/sync-check.sh           # show which pages have drifted from their sources
tools/sync-check.sh --update  # after re-distilling: stamp pages as reconciled at HEAD
```

Each page also names its sources in the HTML itself
(`<meta name="fs-distills" content="docs/00-map.md; …">`) so the link is visible
where you're editing.

## Logo

Working drafts live in [`../assets/logo-drafts/`](../assets/logo-drafts/) (open
`contact-sheet.html` to compare). The chosen marks are copied into
`docs/assets/logo/`: `star.svg` (brand/hero) and `star-mono.svg` (favicon).
