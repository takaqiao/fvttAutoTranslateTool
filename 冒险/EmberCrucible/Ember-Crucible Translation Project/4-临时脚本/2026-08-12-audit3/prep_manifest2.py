#!/usr/bin/env python3
"""第二批（35 本 journal 逐句复核 + U1 + H1 + 补漏）批次的落盘前处理。

做三件事：
  1. 只挑**尚未落盘**的单元（第一批 C1/E*/U2/U3/G*/H2/N1 已在 c33e05b / 17edca2 里）；
  2. 按对抗式复核的 `rejected` 剔除 —— 其中两条是**部分**驳回，同叶别的改动要保留，
     所以做的是精确回退而不是整条丢弃；
  3. 产出 `merge_batches.py` 认得的 manifest。

`lang` 批次不进这里（走 `qa/apply_lang.py`，闸门与键形态都不同）。
"""
from __future__ import annotations
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

S = r'C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3'
B = os.path.join(S, 'batches')
F = os.path.join(S, 'batches_final2')
REPO = {'ember': '1-Ember汉化插件', 'crucible': '2-Crucible汉化插件'}

# 第一批已落盘的单元
APPLIED = {'C1', 'E1', 'E2', 'E3', 'E4', 'E5', 'U2', 'U3', 'G1', 'G2', 'G3', 'G4', 'H2', 'N1'}

# 整条剔除（复核判定原判断是错的）
DROP_WHOLE = {
    ('J08', 'Unfinished Business.pages.The Troubled Caravaneer.text'),
    ('J08', 'Unfinished Business.pages.The Troubled Caravaneer.overview'),
}

# 部分回退：同叶其余改动是对的，只把这一处改回去
# (单元, 路径片段, 批次里的错值, 应回退成)
REVERT = [
    # J04：英文是小写泛指且指的是**战斗追踪器里的 token 名**，
    # `Otherhood Raider.tokenName` 就是「袭击者」。同叶的「离开城市→驶入海洋」是真缺陷，保留。
    ('J04', 'An Old Friend.pages.Dash Away All.text', '一名劫掠者和一名强盗', '一名袭击者和一名强盗'),
    # J08：英文写死了亲属关系（Jorey 是 Agraband 姐妹 Selena 的儿子＝外甥）。
    # 同叶的「乔雷→乔里」两处是对的（Actor.name＝乔里·斯威夫特），保留。
    ('J08', 'Unfinished Business.pages.The Troubled Caravaneer.summary', '他疏远已久的侄子乔里', '他疏远已久的外甥乔里'),
]

os.makedirs(F, exist_ok=True)
man, dropped, reverted = [], [], []
for f in sorted(os.listdir(B)):
    unit, tag, pack = f[:-5].split('__')
    if pack == 'lang' or unit in APPLIED:
        continue
    with open(os.path.join(B, f), encoding='utf-8-sig') as fh:
        items = dict(json.load(fh))
    items = items.get('items', items)

    for u, frag in DROP_WHOLE:
        if u == unit:
            for k in list(items):
                if frag in k:
                    dropped.append((f, k))
                    items.pop(k)
    for u, frag, bad, good in REVERT:
        if u != unit:
            continue
        for k, v in list(items.items()):
            if frag in k and bad in v:
                items[k] = v.replace(bad, good)
                reverted.append((f, k.split('.pages.')[-1], bad, good))

    with open(os.path.join(F, f), 'w', encoding='utf-8') as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    man.append({'kind': 'translations', 'file': os.path.join(F, f),
                'repo': REPO[tag], 'pack': pack + '.json', 'unit': unit})

with open(os.path.join(S, 'manifest2.json'), 'w', encoding='utf-8') as fh:
    json.dump(man, fh, ensure_ascii=False, indent=1)
print('manifest2 条目', len(man))
print('整条剔除：')
for x in dropped:
    print('  ', x[0], '::', x[1].split('.pages.')[-1])
print('部分回退：')
for x in reverted:
    print('  ', x[0], '::', x[1], '|', x[2], '->', x[3])
