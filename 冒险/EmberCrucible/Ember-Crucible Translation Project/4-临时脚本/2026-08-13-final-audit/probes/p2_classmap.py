#!/usr/bin/env python3
"""P2: what class-bearing tags exist, and where do gamemaster/secret live."""
import sys, os, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs

CLS = re.compile(r'<(\w+)[^>]*?\sclass="([^"]*)"')

en_c, cn_c = Counter(), Counter()
fields_gm = Counter()
for rname, repo in REPOS.items():
    for pack, rows in pairs(repo):
        for path, e, c in rows:
            for t, k in CLS.findall(e):
                en_c[f'{t}.{k}'] += 1
            if c:
                for t, k in CLS.findall(c):
                    cn_c[f'{t}.{k}'] += 1
            if 'gamemaster' in e or 'secret' in e.lower():
                leaf = path.split('.')[-1]
                fields_gm[f'{rname}:{leaf}'] += 1

print('=== EN class tokens ===')
for k, v in en_c.most_common(60):
    print(f'{v:>6}  {k}')
print('\n=== CN-only class tokens (not in EN at all) ===')
for k, v in cn_c.most_common():
    if k not in en_c:
        print(f'{v:>6}  {k}')
print('\n=== EN-only class tokens (absent from CN) ===')
for k, v in en_c.most_common():
    if k not in cn_c:
        print(f'{v:>6}  {k}')
print('\n=== count delta EN vs CN (top diffs) ===')
allk = set(en_c) | set(cn_c)
d = sorted(allk, key=lambda k: -abs(en_c[k] - cn_c[k]))
for k in d[:40]:
    if en_c[k] != cn_c[k]:
        print(f'  EN {en_c[k]:>5}  CN {cn_c[k]:>5}   {k}')
print('\n=== leaf fields carrying gamemaster/secret in EN ===')
for k, v in fields_gm.most_common(30):
    print(f'{v:>6}  {k}')
