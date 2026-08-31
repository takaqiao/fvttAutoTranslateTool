#!/usr/bin/env python3
"""P11: the GM-bearing strings P10 could NOT zip 1:1 -- my own blind spot.

P10 compared text nodes pairwise and needed len(EN nodes) == len(CN nodes).
Anything where the counts differ was silently skipped; those are exactly the
strings where a chunk of prose was added, dropped, merged or split -- i.e. the
place a GM paragraph could actually have gone missing or moved. Enumerate them.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK
from htmlblocks import regions

TAGRE = re.compile(r'<[^>]+>')
EMPTY_P = re.compile(r'<p>\s*</p>')
HID = re.compile(r'gamemaster|secret', re.I)


def nodes(html):
    out, pos = [], 0
    for m in TAGRE.finditer(html):
        t = html[pos:m.start()]
        if t.strip():
            out.append((t, pos))
        pos = m.end()
    if html[pos:].strip():
        out.append((html[pos:], pos))
    return out


def spans(html):
    return [(o, e) for cls, i, j, o, e in regions(html, 'section') if HID.search(cls or '')]


def hidlen(html):
    sp = spans(html)
    tot = h = 0
    for t, p in nodes(html):
        n = len(t.strip())
        tot += n
        if any(s <= p < e for s, e in sp):
            h += n
    return h, tot


n_gm = n_zip = 0
short = []
mismatch = []
for rname, repo in REPOS.items():
    for pack, prs in pairs(repo):
        for path, e, c in prs:
            if not (c and CJK.search(c)):
                continue
            e2, c2 = EMPTY_P.sub('', e), EMPTY_P.sub('', c)
            if not spans(e2) and not spans(c2):
                continue
            n_gm += 1
            ne, nc = nodes(e2), nodes(c2)
            if len(ne) != len(nc):
                mismatch.append((rname, pack, path, len(ne), len(nc), e2, c2))
            elif len(ne) < 3:
                short.append((rname, pack, path, len(ne), e2, c2))
            else:
                n_zip += 1

print(f'GM-bearing strings: {n_gm}   zipped by P10: {n_zip}')
print(f'skipped for <3 nodes: {len(short)}   skipped for node-count mismatch: {len(mismatch)}')
print('\n=== node-count mismatches (the real blind spot) ===')
for rname, pack, path, a, b, e, c in mismatch:
    he, te = hidlen(e)
    hc, tc = hidlen(c)
    print(f'\n{rname}/{pack} {path}   EN nodes={a} CN nodes={b}')
    print(f'   EN hidden/total chars = {he}/{te} ({he/max(te,1):.2f})')
    print(f'   CN hidden/total chars = {hc}/{tc} ({hc/max(tc,1):.2f})')
print('\n=== <3 node cases ===')
for rname, pack, path, a, e, c in short[:20]:
    print(f'  {rname}/{pack} {path} nodes={a}')
    print('     EN:', e[:160])
    print('     CN:', c[:160])
