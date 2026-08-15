#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6：块内数字**顺序**互换（多重集相等，判据全瞎）。

scan_content_coverage 比的是多重集，`6 damage for 8 rounds` -> 「8 轮内造成 6 伤害」
它一声不响（6 和 8 都在）。但这两句的规则含义不同：前者是每轮 6 点持续 8 轮，
后者读作 8 轮内共 6 点。**值没错、绑错了单位**，这是数值一致性里最隐蔽的一类。

判据
----
块内 EN 与 CN 的阿拉伯数字**多重集相等**（＝已有判据全绿）但**出现顺序不同**。
只看 2–5 个数字的块（更多的多半是表格，语序噪声大）。

假阳性模式（占绝大多数，必须人工过）
------------------------------------
汉语语序天然会调换：`deals 2d6 damage for 3 rounds` -> 「持续 3 轮，每轮 2d6」、
`a DC 15 check to move 30 feet` 之类的定语后置。所以本探针的产出是**候选清单**，
判定必须回到原句读语义。为压噪：
* 跳过 `d\d`（骰式）内部的数字，只比「独立数值」；
* 跳过两侧顺序只差一次相邻交换、且交换的两个数分属同一单位族的（表格常见）；
  —— 未实现，改为输出「交换距离」供排序，距离越大越可疑。
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import CJK, plain, split_blocks, load_all

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)

DICE = re.compile(r'\b\d*d\d+\b', re.I)
NUM = re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)')


def nums(s: str):
    return NUM.findall(DICE.sub(' ', s))


def inversions(a, b):
    """b 是 a 的一个排列；返回把 b 变回 a 所需的相邻交换数（越大越可疑）。"""
    idx = {}
    for i, v in enumerate(a):
        idx.setdefault(v, []).append(i)
    pos = []
    used = {k: 0 for k in idx}
    for v in b:
        pos.append(idx[v][used[v]])
        used[v] += 1
    inv = 0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if pos[i] > pos[j]:
                inv += 1
    return inv


def page_of(path):
    p = path.split('.')
    return f'{p[1]}/{p[3]}' if len(p) >= 4 and p[2] == 'pages' else '.'.join(p[:2])


def main():
    rows = load_all()
    out = []
    nb = 0
    for repo, pack, path, en, cn in rows:
        if not cn or not CJK.search(cn):
            continue
        eb, cb = split_blocks(en), split_blocks(cn)
        if len(eb) != len(cb):
            continue
        for i, (e, c) in enumerate(zip(eb, cb)):
            pe, pc = plain(e), plain(c)
            if not pe or not CJK.search(pc):
                continue
            a, b = nums(pe), nums(pc)
            if not (2 <= len(a) <= 5) or sorted(a) != sorted(b):
                continue
            nb += 1
            if a == b:
                continue
            out.append({'repo': repo, 'pack': pack, 'path': path,
                        'page': page_of(path), 'block': i,
                        'en_seq': a, 'cn_seq': b, 'inv': inversions(a, b),
                        'en': pe[:400], 'cn': pc[:400]})
    seen, uniq = set(), []
    for r in out:
        k = (r['page'], r['block'], r['en'][:160])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    uniq.sort(key=lambda r: -r['inv'])
    print(f'多重集相等且含 2–5 个独立数值的块：{nb}')
    print(f'其中顺序不同的：{len(out)}（孪生去重后 {len(uniq)}）')
    for r in uniq[:60]:
        print('\n' + '-' * 88)
        print(f'inv={r["inv"]} {r["repo"]} / {r["page"]} [b{r["block"]}] '
              f'{r["en_seq"]} -> {r["cn_seq"]}')
        print('   EN:', r['en'][:300])
        print('   CN:', r['cn'][:300])
    json.dump(uniq, open(os.path.join(OUT, 'xp_p6_num_order.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', os.path.join(OUT, 'xp_p6_num_order.json'))


if __name__ == '__main__':
    main()
