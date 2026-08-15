#!/usr/bin/env python3
"""Locate the exact shipped leaves behind disputes.sameRoleMissingTail.

The bucket is computed from `conflicts_by_role`, which is built ONLY from
`harvest()` -- i.e. from the shipped compendium CN files. The base glossary
never touches it. This script re-runs the same harvest and records, for every
term in the bucket, WHICH FILE and WHICH PATH each candidate came from.
"""
import importlib.util, io, json, os, sys
from collections import Counter, defaultdict

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
BG = os.path.join(P, '3-常用脚本', 'tm', 'build_glossary.py')

# import build_glossary WITHOUT running main()
src = io.open(BG, encoding='utf-8').read().replace('\nmain()\n', '\n')
mod = type(sys)('bg')
mod.__file__ = BG
exec(compile(src, BG, 'exec'), mod.__dict__)

BASE_DIR = os.path.join(P, '5-其他内容', 'english-baseline')
SETS = [
    ('crucible', os.path.join(BASE_DIR, 'crucible-0.10.1'),
     os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn')),
    ('ember', os.path.join(BASE_DIR, 'ember-0.6.0'),
     os.path.join(P, '1-Ember汉化插件', 'compendium', 'cn')),
]

# (en, role) -> cn -> [(label, file, path)]
where = defaultdict(lambda: defaultdict(list))
pairs = defaultdict(Counter)

for label, en_dir, cn_dir in SETS:
    for fn in sorted(f for f in os.listdir(en_dir)
                     if f.endswith('.json') and not f.startswith('_')):
        cn_path = os.path.join(cn_dir, fn)
        if not os.path.exists(cn_path):
            continue
        en_doc = mod.load(os.path.join(en_dir, fn))
        cn_doc = mod.load(cn_path)
        got = []
        mod.walk_pairs(en_doc.get('entries', {}), cn_doc.get('entries', {}), got)
        mod.walk_pairs(en_doc.get('folders', {}), cn_doc.get('folders', {}), got,
                       root_role='folders')
        for en_s, cn_s, path, role in got:
            if not mod.is_term(en_s) or not mod.CJK.search(cn_s):
                continue
            pairs[(en_s, role)][cn_s] += 1
            where[(en_s, role)][cn_s].append((label, fn, path))

d = json.load(io.open(os.path.join(P, '5-其他内容', 'glossary',
                                   'glossary_ec.disputes.json'), encoding='utf-8'))
terms = d['sameRoleMissingTail']['terms']

out = {}
for en_s, per_role in sorted(terms.items()):
    for role, cands in per_role.items():
        rows = {}
        for cn_s in cands:
            rows[cn_s] = [f'{l}:{f} :: {p}' for l, f, p in where[(en_s, role)][cn_s]]
        out[f'{en_s} @{role}'] = rows

o = os.path.join(P, '4-临时脚本', '2026-08-15-round17', 'srmt_locations.json')
json.dump(out, io.open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', o, len(out), 'buckets')
