# `diagram-review` — how to eyeball the site's SVG diagrams

Our diagrams (`docs/**/*.html`) are **hand-authored inline `<svg>`** with absolute
coordinates, colored by CSS variables in `docs/assets/css/main.css`. You cannot
tell whether text overlaps, spills a box, or draws in the wrong order by reading
the numbers — you have to **render and look**, in **both themes**.

[`diagram-review.py`](diagram-review.py) does exactly that: it lifts each
`<figure>` (svg + legend + caption) out of a page, drops it into a throwaway page
that links the *real* stylesheet, and screenshots it with headless Chrome at a
large, predictable size — light and dark.

## Run it

```sh
# every diagram on a page, both themes → .diagram-review/*.png
tools/diagram-review.py docs/learnings/06-mmap.html

# just one figure (0-based, document order)
tools/diagram-review.py docs/learnings/10-transformer-block-anatomy.html --fig 1

# zoom into a region to inspect a join / a tight label
# (viewBox units of THAT svg: "minX minY width height")
tools/diagram-review.py docs/m1-weights.html --fig 0 --zoom "300 20 260 120"
```

Useful flags: `--themes light` (skip dark), `--width 1400` (bigger),
`--scale 3` (sharper), `--out DIR`. Needs Chrome/Chromium/Brave/Edge; set
`$CHROME` to override the binary. Output dir `.diagram-review/` is git-ignored.

## For a coding agent

1. Run the command above for the page you touched.
2. **Read every PNG it prints** — do not skip dark; several bugs only show there
   (translucent fills, low-contrast strokes).
3. Triage against the checklist below. When something's off, `--zoom` into it to
   confirm before editing.
4. Fix in the page's inline SVG (see "How to fix"), then **re-run and re-Read** to
   verify. Check the fix didn't break the other theme.

## What to look for

- **Text spill** — a `<text>` wider than its box, or a left-column label running
  into the next box. Monospace advance ≈ `0.6 × font-size` px per char; estimate
  `chars × 0.6 × size` and compare to the gap. This is the most common bug.
- **Label collisions** — two `<text>` runs, or a label and an arrowhead, sharing
  pixels. Watch stacked arrow labels (above + below a line).
- **Z-order / seams** — edges must be drawn *before* nodes so boxes paint over
  line ends. A line crossing *in front* of a box means it's later in source than
  the box, or the box fill is translucent and the line shows through it.
- **Translucent-fill poke-through** — box fills here are low-opacity
  (`fill-opacity: .16–.24`). A line whose endpoint lands *inside* a box stays
  visible through the fill. Land connectors on the box **edge**, not its center.
- **Clipping** — anything touching or past the `viewBox` edge is cut off. Give it
  margin or widen the `viewBox`.
- **Cramped layout** — boxes/labels with < ~12px of breathing room read as
  crowded even when technically not overlapping.

## How to fix (common moves)

- **More room for a left label:** shift the whole right cluster (boxes, arrows,
  arrow labels) right by N px **and** widen the `viewBox` width by N. Keep every
  element's relative offset identical (this is what the mmap fix did).
- **Line tucks behind a box:** ensure the `<line>`/`<path>` appears **before** the
  `<rect>` in source, and land its endpoint on the box's edge coordinate (e.g.
  right edge `x = rectX + rectWidth`), never past it.
- **Text too wide for its box:** widen the box (and re-center the text at the new
  box center), drop the font-size, or wrap onto two `<text>` lines.
- **Reposition, don't rescale:** nudge coordinates; don't add `transform="scale"`
  — it throws off stroke widths and the coordinate math everyone else reads.

Always confirm the `viewBox` still frames everything with margin after moving
things, and re-render **both** themes.
