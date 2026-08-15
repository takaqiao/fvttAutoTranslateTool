#!/usr/bin/env python3
"""P1: inventory of every string that mentions gamemaster/secret on either side."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK

MARK = re.compile(r'gamemaster|\bsecret\b', re.I)

tot = 0
only_en = []
only_cn = []
both = 0
for rname, repo in REPOS.items():
    for pack, rows in pairs(repo):
        for path, e, c in rows:
            he = bool(MARK.search(e))
            hc = bool(MARK.search(c)) if c else False
            if not he and not hc:
                continue
            tot += 1
            if he and hc:
                both += 1
            elif he and not hc:
                only_en.append((rname, pack, path, e, c))
            else:
                only_cn.append((rname, pack, path, e, c))

print(f'strings mentioning gamemaster/secret: {tot}  both={both}  onlyEN={len(only_en)}  onlyCN={len(only_cn)}')
print('\n=== EN has marker, CN does not (CN present) ===')
n = 0
for rname, pack, path, e, c in only_en:
    if c is None:
        continue
    n += 1
    print(f'\n[{n}] {rname}/{pack}  {path}')
    print('EN:', e[:400].replace('\n', ' '))
    print('CN:', c[:400].replace('\n', ' '))
print(f'\n(EN-marker, CN missing-entirely: {sum(1 for r in only_en if r[4] is None)})')
print('\n=== CN has marker, EN does not ===')
for rname, pack, path, e, c in only_cn:
    print(f'\n{rname}/{pack}  {path}')
    print('EN:', e[:400].replace('\n', ' '))
    print('CN:', (c or '')[:400].replace('\n', ' '))
