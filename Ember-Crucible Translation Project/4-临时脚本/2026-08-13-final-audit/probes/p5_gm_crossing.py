#!/usr/bin/env python3
"""P5: anchors that CROSS the GM boundary.

Refinement of P4: only anchors present on BOTH sides with the SAME total count
are considered, so project-added `id="slug"` anchors (CN-only, 1642 of them, a
known intentional addition) and any add/drop are excluded by construction.
A hit therefore means: the same anchor exists on both sides, but on one side it
sits inside a gamemaster/secret <section> and on the other it does not.
"""
import sys, os, re, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK
from htmlblocks import regions

ANCHOR = re.compile(
    r'@UUID\[[^\]]+\]'
    r'|@Embed\[[^\]]+\]'
    r'|\[\[[^\]]+\]\]'
    r'|src="[^"]+"'
)
HID = re.compile(r'gamemaster|secret', re.I)


def spans(html):
    return [(o, e) for cls, i, j, o, e in regions(html, 'section') if HID.search(cls or '')]


def prof(html):
    sp = spans(html)
    d = defaultdict(lambda: [0, 0])   # anchor -> [n_hidden, n_open]
    for m in ANCHOR.finditer(html):
        h = any(s <= m.start() < e for s, e in sp)
        d[m.group(0)][0 if h else 1] += 1
    return d, sp


hits = []
considered = 0
for rname, repo in REPOS.items():
    for pack, rows in pairs(repo):
        for path, e, c in rows:
            if not (c and CJK.search(c)):
                continue
            de, se_ = prof(e)
            dc, sc_ = prof(c)
            if not se_ and not sc_:
                continue
            for a in set(de) & set(dc):
                he, oe = de[a]
                hc, oc = dc[a]
                if he + oe != hc + oc:
                    continue          # count changed -> not a pure crossing
                considered += 1
                if (he, oe) == (hc, oc):
                    continue
                hits.append({'repo': rname, 'pack': pack, 'path': path,
                             'anchor': a, 'en': [he, oe], 'cn': [hc, oc]})

print(f'comparable anchors in GM-bearing strings: {considered}')
print(f'boundary crossings: {len(hits)}')
for h in hits:
    print(f'  {h["repo"]}/{h["pack"]} {h["path"]}\n     {h["anchor"]}  EN(hid,open)={h["en"]}  CN={h["cn"]}')
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p5_crossing.json')
json.dump(hits, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('->', out)
