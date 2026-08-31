#!/usr/bin/env python3
"""第三批（复核补漏 M* / 补做复核 V* / 孪生包 T1 / 标签↔name L*）批次的落盘前处理。

同 `prep_manifest2.py`，只是 APPLIED 集合往前推了一轮。
`lang` 批次不进这里（走 `qa/apply_lang.py`）。
"""
from __future__ import annotations
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

S = r'C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3'
B = os.path.join(S, 'batches')
F = os.path.join(S, 'batches_final3')
REPO = {'ember': '1-Ember汉化插件', 'crucible': '2-Crucible汉化插件'}

# 已落盘：第三轮 + 第四轮 + 我自己补的三批
APPLIED = {
    'C1', 'E1', 'E2', 'E3', 'E4', 'E5', 'U2', 'U3', 'G1', 'G2', 'G3', 'G4', 'H2', 'N1',   # 第三轮
    'H1', 'H1R', 'N2', 'N3', 'U1', 'U1R',                                                 # 第三轮尾 + 补漏
    *[f'J{i:02d}' for i in range(1, 28)],                                                 # 第四轮 35 本 journal
    *[f'L{i:02d}' for i in range(1, 9)], *[f'M{i:02d}' for i in range(1, 10)], 'N4',       # 第五轮
}

os.makedirs(F, exist_ok=True)
man = []
for f in sorted(os.listdir(B)):
    unit, tag, pack = f[:-5].split('__')
    if pack == 'lang' or unit in APPLIED:
        continue
    with open(os.path.join(B, f), encoding='utf-8-sig') as fh:
        d = json.load(fh)
    items = d.get('items', d)
    with open(os.path.join(F, f), 'w', encoding='utf-8') as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    man.append({'kind': 'translations', 'file': os.path.join(F, f),
                'repo': REPO[tag], 'pack': pack + '.json', 'unit': unit})

with open(os.path.join(S, 'manifest3.json'), 'w', encoding='utf-8') as fh:
    json.dump(man, fh, ensure_ascii=False, indent=1)
print('manifest3 条目', len(man))
for m in man:
    n = len(json.load(open(m['file'], encoding='utf-8')))
    print(f"  {m['unit']:5s} {m['repo'][:1]} {m['pack'][:34]:36s} {n:4d} 条")
