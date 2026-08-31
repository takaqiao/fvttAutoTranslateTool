#!/usr/bin/env python3
"""Classify the 31 `sameRoleMissingTail` buckets against the conventions that
the shipped files ACTUALLY follow, measured leaf by leaf.

Two axes the `(英文, 角色)` key cannot see:
  * MODULE  -- crucible-cn ships `folders` bare (150/151), ember_cn ships it
               bilingual (337/355). Two products, two house styles.
  * SLOT    -- `convention_of` already knows `regions.X.name` is bare while
               `pages.X.name` is bilingual (CONVENTION_PARENTS). The role key
               merges them anyway; the module docstring calls this out as a
               known limit.

A bucket is a GENUINE missing tail only if its two candidates disagree INSIDE
one module and one convention slot. Everything else is two correct populations
being compared across a boundary.
"""
import io, json, os, sys
from collections import Counter, defaultdict

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
R = os.path.join(P, '4-临时脚本', '2026-08-15-round17')
BG = os.path.join(P, '3-常用脚本', 'tm', 'build_glossary.py')
src = io.open(BG, encoding='utf-8').read().replace('\nmain()\n', '\n')
mod = type(sys)('bg'); mod.__file__ = BG
exec(compile(src, BG, 'exec'), mod.__dict__)

BASE_DIR = os.path.join(P, '5-其他内容', 'english-baseline')
SETS = [('crucible', os.path.join(BASE_DIR, 'crucible-0.10.1'),
         os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn')),
        ('ember', os.path.join(BASE_DIR, 'ember-0.6.0'),
         os.path.join(P, '1-Ember汉化插件', 'compendium', 'cn'))]

leaves = []   # (module, fn, path, role, conv, form, en, cn)
for module, en_dir, cn_dir in SETS:
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
            leaves.append((module, fn, path, role,
                           mod.convention_of(path, role),
                           mod.form_of(cn_s, en_s), en_s, cn_s))

d = json.load(io.open(os.path.join(P, '5-其他内容', 'glossary',
                                   'glossary_ec.disputes.json'), encoding='utf-8'))
targets = d['sameRoleMissingTail']['terms']

# index leaves by (en, role)
idx = defaultdict(list)
for L in leaves:
    idx[(L[6], L[3])].append(L)

report = {}
genuine, cross_module, cross_slot = [], [], []
for en_s, per_role in sorted(targets.items()):
    for role in per_role:
        ls = idx[(en_s, role)]
        # candidates inside each (module, conv) cell
        cells = defaultdict(Counter)
        for m, fn, path, r, conv, form, e, c in ls:
            cells[(m, conv)][c] += 1
        split_cells = {k: dict(v) for k, v in cells.items() if len(v) > 1}
        mods = {k[0] for k in cells}
        convs = {k[1] for k in cells}
        if split_cells:
            verdict = 'GENUINE-missing-tail'
            genuine.append(f'{en_s}@{role}')
        elif len(mods) > 1:
            verdict = 'cross-module-convention'
            cross_module.append(f'{en_s}@{role}')
        elif len(convs) > 1:
            verdict = 'cross-slot-convention'
            cross_slot.append(f'{en_s}@{role}')
        else:
            verdict = 'UNEXPLAINED'
            genuine.append(f'{en_s}@{role}')
        report[f'{en_s} @{role}'] = {
            'verdict': verdict,
            'cells': {f'{m}/{cv}': dict(cnt) for (m, cv), cnt in sorted(cells.items())},
            'splitCells': {f'{m}/{cv}': v for (m, cv), v in split_cells.items()},
        }

json.dump({'_summary': {'total': len(report),
                        'GENUINE-missing-tail': len(genuine),
                        'cross-module-convention': len(cross_module),
                        'cross-slot-convention': len(cross_slot)},
           'genuine': genuine, 'crossModule': cross_module, 'crossSlot': cross_slot,
           'detail': report},
          io.open(os.path.join(R, 'classified_31.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('total', len(report), '| genuine', len(genuine),
      '| cross-module', len(cross_module), '| cross-slot', len(cross_slot))
print('genuine:', genuine)
print('crossSlot:', cross_slot)
