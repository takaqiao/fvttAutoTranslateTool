# -*- coding: utf-8 -*-
"""Same EN block -> divergent CN, within the Deities journal."""
import re, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g8_tools import load, blocks, strip
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = load()
m = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    eb, cb = blocks(r["en"]), blocks(r["cn"] or "")
    if not eb:
        eb, cb = [r["en"]], [r["cn"] or ""]
    if len(eb) != len(cb):
        continue
    for i, (e, c) in enumerate(zip(eb, cb)):
        se, sc = strip(e), strip(c)
        if not se:
            continue
        m[se][sc].append(f"{r['k']}#{i}")

n = 0
for se in sorted(m, key=lambda x: -sum(len(v) for v in m[x].values())):
    if len(m[se]) < 2:
        continue
    n += 1
    print(f"\n### EN: {se[:200]}")
    for sc, locs in sorted(m[se].items(), key=lambda kv: -len(kv[1])):
        print(f"   [{len(locs)}] CN: {sc[:200]}")
        print(f"        e.g. {locs[:4]}")
print(f"\ntotal divergent EN strings: {n}")
