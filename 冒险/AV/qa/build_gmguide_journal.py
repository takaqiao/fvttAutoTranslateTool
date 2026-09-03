"""build_gmguide_journal.py - turn the translated GM guide into a Foundry JournalEntry.

Babele cannot carry this content: it translates documents that already exist in a pack,
so a key for a page the pack does not have is an ORPHAN and never renders.  The guide is
external material, so it ships as a standalone JournalEntry JSON that imports into a
world in one drag - no change to the translation module's structure.

Page ids are derived from the content (`gmg` + a base62 SHA1 slice), so re-running this
produces the same ids and a re-import updates in place instead of duplicating.

Usage:
  python build_gmguide_journal.py --pages <translated.pages.json> --out <journal.json>
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# A line this short with no sentence punctuation is a heading, not a paragraph.
HEADING_MAX = 24
SENTENCE_END = re.compile(r"[。！？：]$")
MD_HEADING = re.compile(r"^\s*#{1,6}\s*")
BOLD = re.compile(r"\*\*(.+?)\*\*")
TOC_TAIL = re.compile(r"第\s*\d+\s*页\s*$")
# Pages whose first line is a table header row, which makes a useless page title.
TITLE_OVERRIDES = {4: "支线任务核对清单"}


def base62(digest, length=13):
    n = int.from_bytes(digest, "big")
    out = []
    while n and len(out) < length:
        n, r = divmod(n, 62)
        out.append(B62[r])
    return "".join(out).rjust(length, "0")


def page_id(seed):
    return "gmg" + base62(hashlib.sha1(seed.encode("utf-8")).digest())


def is_heading(line):
    stripped = MD_HEADING.sub("", line).strip()
    if not stripped:
        return False
    if MD_HEADING.match(line):
        return True
    # `赏金猎人……第5页` is a table-of-contents entry, not a section heading.
    if "……" in stripped or TOC_TAIL.search(stripped):
        return False
    return len(stripped) <= HEADING_MAX and not SENTENCE_END.search(stripped) and "，" not in stripped


def to_html(text):
    """Plain translated text -> simple, valid HTML. Markdown leftovers are honoured."""
    out = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        level = 0
        m = MD_HEADING.match(line)
        if m:
            level = min(len(m.group(0).strip()), 4)
        body = MD_HEADING.sub("", line).strip()
        escaped = html.escape(body, quote=False)
        # `**bold**` survives from the PDF pass in a few statblocks.
        escaped = BOLD.sub(r"<strong>\1</strong>", escaped)
        if level:
            out.append(f"<h{max(level, 2)}>{escaped}</h{max(level, 2)}>")
        elif is_heading(line):
            out.append(f"<h3>{escaped}</h3>")
        else:
            out.append(f"<p>{escaped}</p>")
    return "\n".join(out)


def page_title(text, number):
    if number in TITLE_OVERRIDES:
        return f"第 {number} 页 · {TITLE_OVERRIDES[number]}"
    for raw in text.split("\n"):
        line = MD_HEADING.sub("", raw).strip()
        if line:
            return f"第 {number} 页 · {line[:28]}"
    return f"第 {number} 页"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", default="憎恨魔窟 GM 指南 Abomination Vaults GM's Guide")
    args = parser.parse_args(argv)

    source = json.loads(args.pages.read_text(encoding="utf-8"))
    pages = []
    for i, entry in enumerate(source):
        text = (entry.get("translated") or "").strip()
        if not text:
            continue
        number = entry.get("page", i + 1)
        content = to_html(text)
        pages.append({
            "_id": page_id(f"av-gm-guide::{number}"),
            "name": page_title(text, number),
            "type": "text",
            "title": {"show": True, "level": 1},
            "image": {},
            "text": {"format": 1, "content": content},
            "video": {"controls": True, "volume": 0.5},
            "src": None,
            "system": {},
            "sort": (i + 1) * 100000,
            "ownership": {"default": -1},
            "flags": {},
        })

    journal = {
        "name": args.name,
        "pages": pages,
        "folder": None,
        "sort": 0,
        "ownership": {"default": 0},
        "flags": {"av-gm-guide": {"source": "Abomination Vaults GM's Guide, Ron Lundeen, Pathfinder Infinite",
                                   "pages": len(pages)}},
        "_stats": {"systemId": "pf2e", "coreVersion": "14"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(journal, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")

    ids = [p["_id"] for p in pages]
    assert len(set(ids)) == len(ids), "page ids collided"
    total = sum(len(p["text"]["content"]) for p in pages)
    print(f"{len(pages)} pages, {total} chars of HTML -> {args.out}")
    for p in pages[:6]:
        print(f"    {p['_id']}  {p['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
