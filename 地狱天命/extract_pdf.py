"""Extract Hell's Destiny PDF to per-page MD + pages.json + text_reference.txt.

Reads:  C:\\Users\\Taka\\Desktop\\fvtt\\地狱天命\\地狱天命-英文.pdf  (258 pages, 363 MB)
Writes: extracted/source_pages/page_NNN.md  (one per page, ## Page N + text)
        extracted/pages.json                (list of {"page": N, "text": "..."})
        extracted/text_reference.txt        (full dump for grep)
"""
import json
import re
import sys
import io
from pathlib import Path

# Force UTF-8 output (avoid Windows GBK error)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent
PDF = HERE / "地狱天命-英文.pdf"
OUT_DIR = HERE / "extracted"
PAGE_DIR = OUT_DIR / "source_pages"
PAGES_JSON = OUT_DIR / "pages.json"
TEXT_REF = OUT_DIR / "text_reference.txt"

# InDesign soft-hyphen: word broken across line with "-\n" → join (preserve real hyphens)
SOFT_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
# Multiple blank lines → single blank
MULTIBLANK_RE = re.compile(r"\n{3,}")


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = SOFT_HYPHEN_RE.sub(r"\1\2", t)
    t = MULTIBLANK_RE.sub("\n\n", t)
    return t.strip()


def strip_sidebar(text: str) -> str:
    """Strip the chapter ToC navigator sidebar repeated on most content pages.

    The sidebar is a tight block (≤35 lines) starting with 'Campaign Overview'
    and ending with 'Adventure Toolbox'. Pages 4 (Campaign Overview chapter)
    and 224 (Adventure Toolbox chapter) have the same headings but as real
    content — the chapter list is NOT between them, so the ≤35 line bound is safe.
    """
    if not text:
        return text
    lines = text.split("\n")
    start = end = None
    for i, line in enumerate(lines[:60]):
        ls = line.strip()
        if ls == "Campaign Overview" and start is None:
            start = i
        elif ls == "Adventure Toolbox" and start is not None:
            end = i
            break
    if start is not None and end is not None and (end - start) <= 35:
        # Drop the sidebar block (inclusive of both anchor lines)
        return "\n".join(lines[:start] + lines[end + 1 :])
    return text


def main():
    if not PDF.exists():
        print(f"[ERR] PDF not found: {PDF}", file=sys.stderr)
        sys.exit(1)

    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    total = doc.page_count
    print(f"[info] {PDF.name}: {total} pages")

    pages_data = []
    full_text_chunks = []

    sidebar_strip_count = 0
    for i in range(total):
        page = doc[i]
        raw = page.get_text("text")
        text = normalize_text(raw)
        before_len = len(text)
        text = strip_sidebar(text)
        if len(text) < before_len:
            sidebar_strip_count += 1
        n = i + 1
        pages_data.append({"page": n, "text": text, "char_count": len(text)})

        md = f"# Page {n}\n\n{text}\n"
        (PAGE_DIR / f"page_{n:03d}.md").write_text(md, encoding="utf-8")

        full_text_chunks.append(f"=== Page {n} ===\n{text}\n")

        if n % 30 == 0:
            print(f"[info] extracted page {n}/{total}")

    doc.close()

    PAGES_JSON.write_text(
        json.dumps(pages_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    TEXT_REF.write_text("\n".join(full_text_chunks), encoding="utf-8")

    print(f"[done] extracted {total} pages; sidebar stripped on {sidebar_strip_count}")
    print(f"[done] {PAGES_JSON.name}: {PAGES_JSON.stat().st_size:,} bytes")
    print(f"[done] {TEXT_REF.name}: {TEXT_REF.stat().st_size:,} bytes")
    print(f"[done] {PAGE_DIR.name}: {len(list(PAGE_DIR.glob('page_*.md')))} files")

    avg_chars = sum(p["char_count"] for p in pages_data) / total
    short_pages = sorted(pages_data, key=lambda p: p["char_count"])[:5]
    long_pages = sorted(pages_data, key=lambda p: p["char_count"], reverse=True)[:5]
    print(f"[info] avg chars/page: {avg_chars:.0f}")
    print(f"[info] shortest 5: {[(p['page'], p['char_count']) for p in short_pages]}")
    print(f"[info] longest 5: {[(p['page'], p['char_count']) for p in long_pages]}")


if __name__ == "__main__":
    main()
