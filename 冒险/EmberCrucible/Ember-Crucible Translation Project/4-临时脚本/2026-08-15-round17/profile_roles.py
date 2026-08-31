#!/usr/bin/env python3
"""Profile the FORM (bare / bilingual) of every harvested leaf, sliced by the
things `role_of` throws away: the source file and the container path.

Answers the only question that matters for `sameRoleMissingTail`: is the 31 a
population of leaves that broke ONE convention, or a population of leaves that
belong to TWO different conventions the role key cannot see?
"""
import io, json, os, sys
from collections import Counter, defaultdict

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
BG = os.path.join(P, '3-常用脚本', 'tm', 'build_glossary.py')
src = io.open(BG, encoding='utf-8').read().replace('\nmain()\n', '\n')
mod = type(sys)('bg'); mod.__file__ = BG
exec(compile(src, BG, 'exec'), mod.__dict__)

BASE_DIR = os.path.join(P, '5-其他内容', 'english-baseline')
SETS = [('crucible', os.path.join(BASE_DIR, 'crucible-0.10.1'),
         os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn')),
        ('ember', os.path.join(BASE_DIR, 'ember-0.6.0'),
         os.path.join(P, '1-Ember汉化插件', 'compendium', 'cn'))]

# slice -> Counter(form)
by_file_folders = defaultdict(Counter)     # folders role, per source file
by_slot = defaultdict(Counter)             # role + structural slot
rows = []

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
            form = mod.form_of(cn_s, en_s)
            conv = mod.convention_of(path, role)
            segs = path.split('.')
            # structural slot: is this a TOP-LEVEL pack folder (one segment,
            # came from the `folders` block) or a folder nested inside an entry?
            if role == 'folders':
                slot = 'pack-folder' if len(segs) == 1 else 'entry-folder'
                by_file_folders[f'{label}:{fn} [{slot}]'][form] += 1
            else:
                slot = '.'.join(s for s in segs[:-1] if not s.isdigit())
                slot = 'CONVPARENT' if set(segs) & mod.CONVENTION_PARENTS else 'plain'
            by_slot[(role, slot, conv)][form] += 1
            rows.append((label, fn, path, role, slot, conv, form, en_s, cn_s))

o = os.path.join(P, '4-临时脚本', '2026-08-15-round17')
with io.open(os.path.join(o, 'profile_folders_by_file.txt'), 'w', encoding='utf-8') as f:
    for k in sorted(by_file_folders):
        c = by_file_folders[k]
        f.write(f'{k:70s} bare={c["bare"]:4d} bilingual={c["bilingual"]:4d}\n')
with io.open(os.path.join(o, 'profile_by_slot.txt'), 'w', encoding='utf-8') as f:
    for k in sorted(by_slot, key=lambda k: -sum(by_slot[k].values())):
        c = by_slot[k]
        f.write(f'{str(k):60s} bare={c["bare"]:5d} bilingual={c["bilingual"]:5d}\n')

# --- the real test: re-bucket conflicts on (en, role, slot) --------------
pairs3 = defaultdict(Counter)
for label, fn, path, role, slot, conv, form, en_s, cn_s in rows:
    key = (en_s, role, slot)
    pairs3[key][cn_s] += 1

def bare_(cn, en):
    return cn[:-len(en)].strip() if cn.endswith(en) else cn

srmt3, ti3 = {}, {}
for (en_s, role, slot), cands in pairs3.items():
    if len(cands) < 2:
        continue
    if len({bare_(c, en_s) for c in cands}) == 1:
        srmt3.setdefault(en_s, {})[f'{role}@{slot}'] = dict(cands.most_common())
    else:
        ti3.setdefault(en_s, {})[f'{role}@{slot}'] = dict(cands.most_common())

json.dump({'sameRoleMissingTail_sliced': srmt3, 'termInconsistency_sliced': ti3},
          io.open(os.path.join(o, 'resliced.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('sliced sameRoleMissingTail:', len(srmt3))
print('sliced termInconsistency  :', len(ti3))
