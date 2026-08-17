# -*- coding: utf-8 -*-
"""P17: 改动桶 334 叶的三分类 + 机制变动名单；删除桶 314 叶的迁移/真删名单。"""
import os, sys, json, difflib, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
MECH = re.compile(r'\[\[/[a-z]+|@Condition\[|@Advantage\[|&(?:amp;)?[Rr]eference\[|@Action\[|@Spell\[|@UUID\[|\b\d+\s?(?:ft|feet|gp|sp|cp|d\d+)\b|\bDC\b')
rows = []
for f in FILES:
    for ptr, (old, new) in delta[f]['changed'].items():
        sm = difflib.SequenceMatcher(None, old, new)
        ops = set(t[0] for t in sm.get_opcodes() if t[0] != 'equal')
        kind = '纯增补' if ops <= {'insert'} else ('纯删减' if ops <= {'delete'} else '改写')
        # 机制变动：被改掉/新加的片段里出现规则记号
        mech = False
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            if MECH.search(old[i1:i2]) or MECH.search(new[j1:j2]):
                mech = True
                break
        r = sm.ratio()
        rows.append({'file': f, 'ptr': ptr, 'kind': kind, 'mech': mech, 'ratio': round(r, 4),
                     'len': [len(old), len(new)]})

c = collections.Counter(x['kind'] for x in rows)
print('改动桶 334 叶三分类：', dict(c), '合计', sum(c.values()))
m = [x for x in rows if x['mech']]
print('其中触及规则记号（机制/数值/增强器 改动）的：', len(m))
big = [x for x in rows if x['ratio'] < 0.5]
print('相似度 < 0.5（整段重译）：', len(big))
mid = [x for x in rows if 0.5 <= x['ratio'] < 0.8]
print('相似度 0.5–0.8（大幅改写）：', len(mid))
sml = [x for x in rows if x['ratio'] >= 0.8]
print('相似度 >= 0.8（局部补丁）：', len(sml))
jdump(rows, os.path.join(WORK, 'changed_classified.json'))

print('\n=== 机制/数值改动逐条（去重后按页） ===')
seen = set()
for x in sorted(m, key=lambda y: y['ptr']):
    k = x['ptr']
    if k in seen:
        continue
    seen.add(k)
    print(f"  [{x['kind']}] r={x['ratio']:.3f} {k.split('/',3)[-1]}")
print(f'  （去重后 {len(seen)} 条路径）')
