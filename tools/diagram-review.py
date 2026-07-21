#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# diagram-review.py — render the site's inline SVG diagrams to PNGs so a
# (human or coding-agent) reviewer can actually *see* them.
#
# Why this exists: our diagrams are hand-authored inline <svg> with absolute
# coordinates, styled by CSS variables from assets/css/main.css. You cannot
# eyeball overlap / text-spill / z-order bugs by reading coordinates. This tool
# extracts each <figure> (or standalone role="img" svg), drops it into a minimal
# page that links the *real* stylesheet, and screenshots it with headless Chrome
# in BOTH themes at a large, predictable size.
#
#   tools/diagram-review.py docs/learnings/06-mmap.html
#   tools/diagram-review.py docs/learnings/10-transformer-block-anatomy.html --fig 1
#   tools/diagram-review.py docs/m1-weights.html --fig 0 --zoom "300 20 260 120"
#
# Output: PNGs under .diagram-review/ (git-ignored) + a printed manifest.
# Deps: python3 stdlib only + a Chromium-family browser (Chrome/Chromium/Brave/Edge).
#
# See tools/diagram-review.md for the reviewer's checklist (what to look for and
# how to fix it).
# ---------------------------------------------------------------------------
import argparse
import math
import os
import re
import shutil
import subprocess
import sys

REPO = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()

CHROME_CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chromium-browser") or "",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if c and os.path.exists(c):
            return c
    sys.exit(
        "No Chromium-family browser found. Install Chrome, or set $CHROME to the binary."
    )


def extract_figures(html):
    """Return a list of (kind, inner_html) diagram blocks in document order.

    We take whole <figure> blocks (they carry the svg + legend + caption, all of
    which can overlap), plus any standalone role=\"img\" <svg> not already inside
    a figure. The theme-toggle icon svgs (no role=\"img\") are ignored.
    """
    figs = re.findall(r"<figure\b.*?</figure>", html, re.S)
    joined = "".join(figs)
    svgs = re.findall(r"<svg\b.*?</svg>", html, re.S)
    standalone = [s for s in svgs if s not in joined and 'role="img"' in s]
    return [("figure", f) for f in figs] + [("svg", s) for s in standalone]


def viewbox_aspect(block):
    """Height/width ratio from the first viewBox in the block (fallback 0.5)."""
    m = re.search(r'viewBox="\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', block)
    if not m:
        return 0.5
    _, _, w, h = (float(x) for x in m.groups())
    return h / w if w else 0.5


def css_href(html, html_path):
    m = re.search(r'href="([^"]*main\.css)"', html)
    if not m:
        sys.exit("Could not find a main.css <link> in the page.")
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(html_path)), m.group(1)))


def build_page(block, css_abs, theme, width, zoom):
    if zoom:
        block = re.sub(
            r'(viewBox=")\s*[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+(")',
            r"\g<1>" + zoom + r"\g<2>",
            block,
            count=1,
        )
    theme_attr = ' data-theme="light"' if theme == "light" else ' data-theme="dark"'
    return f"""<!DOCTYPE html><html{theme_attr}><head><meta charset="utf-8">
<link rel="stylesheet" href="file://{css_abs}">
<style>
  body {{ margin: 0; padding: 20px; }}
  .diagbox {{ width: {width}px; border: 1px dashed #d33; padding: 10px; }}
  .diaglabel {{ font: 12px monospace; color: #d33; margin: 0 0 6px; }}
  .diagbox svg {{ width: 100% !important; height: auto !important; }}
</style></head><body>
<div class="diagbox"><p class="diaglabel">{{LABEL}}</p>{block}</div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Render site SVG diagrams to PNGs for review.")
    ap.add_argument("html", help="path to an HTML page under docs/")
    ap.add_argument("--out", default=os.path.join(REPO, ".diagram-review"), help="output dir")
    ap.add_argument("--themes", default="light,dark", help="comma list: light,dark")
    ap.add_argument("--width", type=int, default=1100, help="render width in CSS px")
    ap.add_argument("--fig", type=int, default=None, help="only this figure index (0-based)")
    ap.add_argument("--zoom", default=None, help='override viewBox: "x y w h" (needs --fig)')
    ap.add_argument("--scale", type=int, default=2, help="device scale factor")
    ap.add_argument("--extra", type=int, default=220, help="px added below the svg for legend/caption")
    args = ap.parse_args()

    if args.zoom and args.fig is None:
        sys.exit("--zoom requires --fig (a single figure to zoom into).")

    chrome = find_chrome()
    html_path = args.html
    with open(html_path) as f:
        html = f.read()

    blocks = extract_figures(html)
    if not blocks:
        sys.exit(f"No diagram figures/svgs found in {html_path}.")
    css_abs = css_href(html, html_path)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(html_path))[0]
    indices = [args.fig] if args.fig is not None else range(len(blocks))
    themes = [t.strip() for t in args.themes.split(",") if t.strip()]

    manifest = []
    for i in indices:
        kind, block = blocks[i]
        # height: svg fills the fixed width, so its height is width*aspect; add
        # headroom for the legend + caption that sit below it inside the figure.
        aspect = viewbox_aspect(block if not args.zoom else f'viewBox="{args.zoom}"')
        win_h = math.ceil(args.width * aspect) + args.extra + 80
        for theme in themes:
            page = build_page(block, css_abs, theme, args.width, args.zoom).replace(
                "{LABEL}", f"{stem}  ·  figure {i} ({kind})  ·  {theme}"
                + (f"  ·  zoom [{args.zoom}]" if args.zoom else "")
            )
            zsfx = "-zoom" if args.zoom else ""
            html_out = os.path.join(args.out, f"{stem}-fig{i}-{theme}{zsfx}.html")
            png_out = os.path.join(args.out, f"{stem}-fig{i}-{theme}{zsfx}.png")
            with open(html_out, "w") as f:
                f.write(page)
            subprocess.run(
                [
                    chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
                    f"--force-device-scale-factor={args.scale}",
                    f"--window-size={args.width + 60},{win_h}",
                    f"--screenshot={png_out}",
                    f"file://{html_out}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            manifest.append((i, kind, theme, png_out))

    print(f"\n{len(manifest)} render(s) → {args.out}\n")
    for i, kind, theme, png in manifest:
        print(f"  fig {i:<2} {theme:<5} {kind:<6} {png}")
    print("\nOpen the PNGs (Read tool for an agent) and check against tools/diagram-review.md.")


if __name__ == "__main__":
    main()
