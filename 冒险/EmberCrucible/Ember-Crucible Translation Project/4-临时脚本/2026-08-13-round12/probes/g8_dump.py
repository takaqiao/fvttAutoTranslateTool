# -*- coding: utf-8 -*-
"""Write block-aligned EN/CN dumps for the Deities journal, chunked by page."""
import os, sys, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g8_tools import load, blocks, pagename
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "read")
os.makedirs(OUT, exist_ok=True)

rows = load()
bypage = collections.OrderedDict()
for r in rows:
    p, f = pagename(r["k"])
    if p is None:
        p = "__meta__"
    bypage.setdefault(p, []).append(r)

FIELD_ORDER = {"name": 0, "pronunciation": 1, "subtitle": 2, "bannerCaption": 3,
               "contentOverview": 4, "text": 5, "contentGamemaster": 6}

pages = list(bypage.items())
CHUNK_CHARS = 95000
chunk, size, idx = [], 0, 1
files = []


def flush():
    global chunk, size, idx
    if not chunk:
        return
    fn = os.path.join(OUT, f"g8_read_{idx:02d}.txt")
    open(fn, "w", encoding="utf-8").write("\n".join(chunk))
    files.append(fn)
    chunk, size, idx = [], 0, idx + 1


for p, rs in pages:
    buf = [f"\n\n================ PAGE: {p} ================"]
    rs.sort(key=lambda r: FIELD_ORDER.get(pagename(r["k"])[1], 9))
    for r in rs:
        f = pagename(r["k"])[1]
        eb, cb = blocks(r["en"]), blocks(r["cn"] or "")
        buf.append(f"\n---- field {f}  (enBlk={len(eb)} cnBlk={len(cb)})")
        if not eb:
            buf.append(f"E| {r['en']}")
            buf.append(f"C| {r['cn']}")
            continue
        for i in range(max(len(eb), len(cb))):
            e = eb[i] if i < len(eb) else "<<<MISSING>>>"
            c = cb[i] if i < len(cb) else "<<<MISSING>>>"
            buf.append(f"{i}E| {e}")
            buf.append(f"{i}C| {c}")
    t = "\n".join(buf)
    if size + len(t) > CHUNK_CHARS and chunk:
        flush()
    chunk.append(t)
    size += len(t)
flush()
for f in files:
    print(f, os.path.getsize(f))
