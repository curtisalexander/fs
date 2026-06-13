#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sync-check.sh — drift detector for the HTML site.
#
# Our model: the markdown in docs/ is the WORKING COPY (source of truth). The
# HTML in docs/*.html is a hand-authored DISTILLATION — intentionally not a
# word-for-word conversion. To keep the two "kinda sorta in sync" we don't
# auto-generate anything; instead each page declares which markdown it distills
# (tools/sync-ledger.tsv), recording the commit at which they were last
# reconciled. This script reports when a source has new commits since — i.e.
# when a published page is due for a re-read and re-distill.
#
#   tools/sync-check.sh          # report which pages have drifted from sources
#   tools/sync-check.sh --update # stamp every page as reconciled at current HEAD
#
# Ledger format (tab-separated, '#' lines ignored):
#   <html_path>\t<comma,separated,sources>\t<synced_commit>
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LEDGER="tools/sync-ledger.tsv"
MODE="${1:-report}"
HEAD_SHA="$(git rev-parse --short HEAD)"

[ -f "$LEDGER" ] || { echo "no ledger at $LEDGER"; exit 1; }

if [ "$MODE" = "--update" ]; then
  # Re-stamp every data row's synced_commit to current HEAD, preserving comments.
  tmp="$(mktemp)"
  while IFS= read -r line; do
    case "$line" in
      \#*|"") printf '%s\n' "$line" ;;
      *) html="${line%%	*}"; rest="${line#*	}"; srcs="${rest%%	*}"
         printf '%s\t%s\t%s\n' "$html" "$srcs" "$HEAD_SHA" ;;
    esac
  done < "$LEDGER" > "$tmp"
  mv "$tmp" "$LEDGER"
  echo "stamped all pages as reconciled at $HEAD_SHA"
  exit 0
fi

drift=0
while IFS=$'\t' read -r html srcs synced; do
  case "$html" in \#*|"") continue ;; esac
  for src in ${srcs//,/ }; do
    if [ "$synced" = "PENDING" ] || ! git cat-file -e "$synced^{commit}" 2>/dev/null; then
      printf '  ? %-22s <- %-28s (no baseline; run --update)\n' "$html" "$src"
      drift=1; continue
    fi
    n="$(git rev-list --count "${synced}..HEAD" -- "$src" 2>/dev/null || echo 0)"
    if [ "$n" -gt 0 ]; then
      printf '  ! %-22s <- %-28s (%s new commit(s) since %s)\n' "$html" "$src" "$n" "$synced"
      drift=1
    else
      printf '  ✓ %-22s <- %-28s\n' "$html" "$src"
    fi
  done
done < "$LEDGER"

echo
if [ "$drift" -eq 0 ]; then
  echo "in sync — every page matches its sources at the recorded commit."
else
  echo "drift found. Re-read the source(s), update the page if needed, then:"
  echo "    tools/sync-check.sh --update"
fi
exit 0
