"""On-demand lookup against the offline pf2 wiki mirror (_wiki_full_v2/).

Tier-4 fallback for terms missed by:
  1. state/02_term_map.json (AP-specific)
  2. lookup_term.py against tm_3source.json (SRD)

Usage:
    python lookup_wiki.py "Order of the Manacle"
    python lookup_wiki.py "Twilight Talons" --english-only

For each query:
  - Searches page filenames (Chinese pages often named with the canonical Chinese)
  - Greps content of matching pages for English keyword + Chinese context
  - Returns top 3 matches with snippet

Uses ripgrep if available, else grep, else python re fallback.
"""
import argparse
import re
import subprocess
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
WIKI = HERE.parent / "_wiki_full_v2"
PAGES = WIKI / "pages"
CATEGORY = WIKI / "category"

# Strip HTML tags + collapse whitespace for snippet display
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_snippet(s: str, max_len: int = 300) -> str:
    s = TAG_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s[:max_len]


def find_files_by_filename(term: str) -> list[Path]:
    """Quick: find HTML files whose name contains the term (Chinese or English)."""
    if not PAGES.exists():
        return []
    matches = []
    term_lc = term.lower()
    for p in PAGES.iterdir():
        if not p.is_file() or p.suffix != ".html":
            continue
        if term_lc in p.name.lower():
            matches.append(p)
    return sorted(matches, key=lambda p: len(p.name))[:5]


def grep_content(term: str, max_hits: int = 5) -> list[tuple[Path, str]]:
    """Search file content. Use grep -l to find files, then snippet."""
    if not PAGES.exists():
        return []
    try:
        # grep -lr: list files with match
        out = subprocess.run(
            ["grep", "-lir", "--include=*.html", term, str(PAGES)],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
            timeout=30,
        )
        files = [Path(p) for p in out.stdout.strip().splitlines() if p][:max_hits]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        files = []
    results = []
    for fp in files:
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
            # Find first occurrence and grab surrounding context
            idx = content.lower().find(term.lower())
            if idx == -1:
                continue
            start = max(0, idx - 100)
            end = min(len(content), idx + 200)
            snippet = clean_snippet(content[start:end])
            results.append((fp, snippet))
        except Exception:
            continue
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("term", help="English or Chinese term to look up")
    ap.add_argument("--snippets", type=int, default=3, help="max snippets to show")
    args = ap.parse_args()

    term = args.term
    print(f"=== Lookup: {term!r} ===")

    # 1. Filename matches
    fn_hits = find_files_by_filename(term)
    if fn_hits:
        print(f"\n[filenames] {len(fn_hits)} match(es):")
        for p in fn_hits:
            print(f"  - {p.name}")

    # 2. Content matches
    ct_hits = grep_content(term, max_hits=args.snippets)
    if ct_hits:
        print(f"\n[content] {len(ct_hits)} match(es):")
        for p, snip in ct_hits:
            print(f"  --- {p.name} ---")
            print(f"    {snip}")
    elif not fn_hits:
        print("\n(no matches)")


if __name__ == "__main__":
    main()
