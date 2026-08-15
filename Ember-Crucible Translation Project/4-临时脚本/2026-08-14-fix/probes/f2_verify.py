# -*- coding: utf-8 -*-
"""Post-build verification: show every changed span, and prove twin coverage."""
import json, os, re, difflib
import f2_lib as L

OUT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-14-fix\batches"
MAP = {'F2.1.': L.EMBER, 'F2.2.': L.CRUC}

batches = {}
for fn in sorted(os.listdir(OUT)):
    if not fn.startswith('F2.'):
        continue
    pre = fn[:5]
    pack = fn[5:]
    repo = MAP[pre]
    with open(os.path.join(OUT, fn), encoding='utf-8') as f:
        batches[(repo, pack)] = json.load(f)

print('########## CHANGED SPANS')
for (repo, pack), b in batches.items():
    cn = L.cnmap(repo, pack)
    for p, nv in b.items():
        ov = cn[p]
        sm = difflib.SequenceMatcher(None, ov, nv, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            print(f'{pack} | {p}')
            print(f'    -{ov[max(0,i1-18):i2+18]!r}')
            print(f'    +{nv[max(0,j1-18):j2+18]!r}')

print()
print('########## TWIN COVERAGE (ember.adventure <-> ember.crucible-adventure)')
a = 'ember.adventure.json'; c = 'ember.crucible-adventure.json'
ena = L.enmap(L.EMBER, a); enc = L.enmap(L.EMBER, c)
ba = batches.get((L.EMBER, a), {}); bc = batches.get((L.EMBER, c), {})
for p in sorted(set(ba) | set(bc)):
    ina, inc = p in ba, p in bc
    if ina and inc:
        continue
    other = c if ina else a
    eo = (enc if ina else ena).get(p)
    es = (ena if ina else enc).get(p)
    same = (eo == es)
    print(f'  only in {a if ina else c}: {p}')
    print(f'      leaf exists in {other} EN? {eo is not None}   EN byte-identical? {same}')
