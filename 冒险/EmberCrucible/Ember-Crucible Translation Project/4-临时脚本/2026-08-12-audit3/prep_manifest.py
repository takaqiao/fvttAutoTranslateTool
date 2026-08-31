#!/usr/bin/env python3
"""把审计三各单元的批次整理成 merge_batches.py 认得的 manifest，并剔除复核驳回项。

批次文件名编码了仓库与包：`<单元>__<ember|crucible>__<包名去.json>.json`。
`lang` 那两个不进这里 —— lang 走 `qa/apply_lang.py`，闸门与键形态都不同。
"""
from __future__ import annotations
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

S = r'C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3'
B = os.path.join(S, 'batches')
F = os.path.join(S, 'batches_final')
REPO = {'ember': '1-Ember汉化插件', 'crucible': '2-Crucible汉化插件'}

# 对抗式复核驳回的条目（理由见 findings/审计三报告）：
# U3 用低阶证据推翻了最强的 name 字段层，且与自己对 #202/#203 的裁决自相矛盾。
DROP = [('U3', 'Local Color.pages.Matters of Perspec')]

os.makedirs(F, exist_ok=True)
man, dropped = [], []
for f in sorted(os.listdir(B)):
    unit, tag, pack = f[:-5].split('__')
    if pack == 'lang':
        continue
    with open(os.path.join(B, f), encoding='utf-8-sig') as fh:
        d = json.load(fh)
    items = dict(d.get('items', d))
    for u, frag in DROP:
        if u == unit:
            for k in list(items):
                if frag in k:
                    dropped.append((f, k))
                    items.pop(k)
    with open(os.path.join(F, f), 'w', encoding='utf-8') as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    man.append({'kind': 'translations', 'file': os.path.join(F, f),
                'repo': REPO[tag], 'pack': pack + '.json', 'unit': unit})

with open(os.path.join(S, 'manifest.json'), 'w', encoding='utf-8') as fh:
    json.dump(man, fh, ensure_ascii=False, indent=1)
print('manifest 条目', len(man))
print('按驳回剔除:', dropped)
