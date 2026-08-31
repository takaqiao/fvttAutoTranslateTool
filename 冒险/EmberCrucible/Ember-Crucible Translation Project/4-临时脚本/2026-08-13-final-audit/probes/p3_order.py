#!/usr/bin/env python3
"""P3: ordered tag.class sequence EN vs CN.

Multiset equality is already 0 project-wide (scan_class_drift). This probe asks a
strictly stronger question: is the class-bearing tag sequence in the SAME ORDER?
A reorder with equal multiset = a class that now wraps a different block.
False-positive mode: legitimate reordering of two sibling blocks of DIFFERENT
class in translation (rare in this corpus; we print every hit for eyeball).
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK

CLS = re.compile(r'<(\w+)[^>]*?\sclass="([^"]*)"')

hits = []
tot = 0
for rname, repo in REPOS.items():
    for pack, rows in pairs(repo):
        for path, e, c in rows:
            if not (c and CJK.search(c)):
                continue
            se = [f'{t}.{k}' for t, k in CLS.findall(e)]
            sc = [f'{t}.{k}' for t, k in CLS.findall(c)]
            if not se and not sc:
                continue
            tot += 1
            if se == sc:
                continue
            hits.append({'repo': rname, 'pack': pack, 'path': path,
                         'en_seq': se, 'cn_seq': sc,
                         'same_multiset': sorted(se) == sorted(sc),
                         'en': e, 'cn': c})

print(f'class-bearing translated strings: {tot}')
print(f'sequence mismatch: {len(hits)}')
print(f'  of which SAME multiset (pure reorder / re-attach): '
      f'{sum(1 for h in hits if h["same_multiset"])}')
for h in hits:
    print('\n' + '=' * 70)
    print(f'{h["repo"]}/{h["pack"]}  {h["path"]}  same_multiset={h["same_multiset"]}')
    print('EN seq:', h['en_seq'])
    print('CN seq:', h['cn_seq'])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p3_order.json')
json.dump(hits, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n->', out)
