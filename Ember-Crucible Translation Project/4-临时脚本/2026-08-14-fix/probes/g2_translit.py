"""G2 finding #2: locate every leaf carrying each translit variant, with counts.

Prints, per variant pair, the leaves that hold the minority spelling so the
batch can rewrite whole leaves.
"""
import sys
import json
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_lib import R1, R2, pack_leaves, all_packs  # noqa: E402

PAIRS = [
    ('Nathira', '纳西拉', '纳希拉'),
    ('Gedron', '盖德隆', '格德隆'),
    ('Gurty', '格蒂', '古蒂'),
    ('Tyrwar', '提尔瓦', '提尔沃'),
    ('Sielle', '希耶尔', '西耶尔'),
    ('Penni', '彭妮', '佩妮'),
    ('Kohle', '科代恩·科勒', '科代恩·科尔'),
    ('Sarinland', '萨林兰', '萨林兰德'),
]

res = {}
for repo, pack in all_packs():
    eo, co = pack_leaves(repo, pack)
    for en_tok, a, b in PAIRS:
        for k, cn in co.items():
            na, nb = cn.count(a), cn.count(b)
            if en_tok == 'Sarinland':
                na = cn.count('萨林兰') - cn.count('萨林兰德')
            if na or nb:
                res.setdefault(en_tok, []).append({
                    'pack': pack, 'path': k, a: na, b: nb,
                    'en_has': eo.get(k, '').count(en_tok),
                })

for tok, rows in res.items():
    a, b = [p for p in PAIRS if p[0] == tok][0][1:]
    ta = sum(r[a] for r in rows)
    tb = sum(r[b] for r in rows)
    print(f'=== {tok}: {a}={ta}  {b}={tb}')
    for r in rows:
        if r[a] and r[b]:
            mark = 'BOTH'
        elif r[a]:
            mark = a
        else:
            mark = b
        print(f'   [{mark}] {r["pack"]} :: {r["path"]}  ({a}={r[a]}, {b}={r[b]}, en_tok={r["en_has"]})')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'g2_translit.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
