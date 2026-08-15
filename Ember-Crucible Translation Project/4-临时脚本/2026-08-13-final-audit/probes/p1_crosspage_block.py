#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：**块级**跨页同英文 -> 中文数值/单位是否自相矛盾。

现有 scan_same_en_split 以「整叶英文全串」为键，只能看见整叶重复；
本探针把每个叶子按块级标签切开、EN/CN 按下标对齐，以**单个块**的英文为键，
于是「同一条规则被抄进 12 个不同页面的某一段里」这种情形第一次可见。

只报**数值或机制单位**层面的分歧（措辞变体不报），因为措辞变体在本库合法。

假阳性模式（读结果必须知道）
----------------------------
1. 对齐依赖 EN/CN 块数相等。不等的叶子被跳过（输出里有 skipped 计数）＝假阴性，不是假阳性。
2. 同英文块在不同语境下**本来就该译得不同**（同形异义），这时数字一般相同、
   只有单位词不同 —— 所以单位分歧一栏需要人工看语境。
3. 孪生包（ember.adventure / ember.crucible-adventure）里同一 (journal,page) 的
   两份内容视为**同一处**，不算跨页；跨页要求 (journal,page) 不同。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CJK, NUM, plain, split_blocks, load_all, cn_num_value, CN_NUM)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)

# 机制单位：英文侧关键词 -> 中文侧可接受写法（用于「同一机制不同页单位是否统一」）
UNITS = {
    'feet': ['英尺', '尺'],
    'round': ['轮'],
    'turn': ['回合'],
    'action': ['动作'],
    'focus': ['专注'],
    'hour': ['小时'],
    'minute': ['分钟'],
    'day': ['天', '日'],
}
UNIT_CN_ALL = ['英尺', '尺', '轮', '回合', '动作', '专注', '小时', '分钟', '天', '日',
               '点', '级', '环', '格', '码', '英里', '里']


def cn_numbers(s: str):
    """中文侧数字多重集：阿拉伯数字 + 可折算的中文数词。"""
    c = Counter(NUM.findall(s))
    for m in CN_NUM.finditer(s):
        v = cn_num_value(m.group())
        if v is not None:
            c[str(v)] += 1
    return c


def unit_profile(s: str):
    return tuple(sorted(u for u in UNIT_CN_ALL if u in s))


def norm_en(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def page_of(path: str) -> str:
    """entries.<Journal>.pages.<Page>.text... -> '<Journal>/<Page>'；其它取前两段。"""
    parts = path.split('.')
    if len(parts) >= 4 and parts[2] == 'pages':
        return f'{parts[1]}/{parts[3]}'
    return '.'.join(parts[:2])


def main():
    rows = load_all()
    groups = defaultdict(list)   # norm_en_block -> [(repo,pack,path,idx,en,cn)]
    skipped = 0
    aligned = 0
    for repo, pack, path, en, cn in rows:
        if not cn or not CJK.search(cn):
            continue
        eb, cb = split_blocks(en), split_blocks(cn)
        if len(eb) != len(cb):
            skipped += 1
            continue
        aligned += 1
        for i, (e, c) in enumerate(zip(eb, cb)):
            pe, pc = plain(e), plain(c)
            if len(pe) < 24 or not CJK.search(pc):
                continue
            if not (NUM.search(pe)):
                continue
            groups[norm_en(pe)].append((repo, pack, path, i, pe, pc))

    findings = []
    for key, occ in groups.items():
        if len(occ) < 2:
            continue
        pages = {page_of(o[2]) for o in occ}
        # 数值分歧
        by_num = defaultdict(list)
        by_unit = defaultdict(list)
        for o in occ:
            by_num[tuple(sorted(cn_numbers(o[5]).items()))].append(o)
            by_unit[unit_profile(o[5])].append(o)
        num_split = len(by_num) > 1
        unit_split = len(by_unit) > 1
        if not (num_split or unit_split):
            continue
        # 跨页要求：分歧出现在不同 (journal,page)
        def spread(d):
            ps = [{page_of(o[2]) for o in v} for v in d.values()]
            return len(set.union(*ps)) > 1 if ps else False
        if num_split and not spread(by_num):
            num_split = False
        if unit_split and not spread(by_unit):
            unit_split = False
        if not (num_split or unit_split):
            continue
        findings.append({
            'en': key[:400],
            'en_nums': sorted(Counter(NUM.findall(key)).items()),
            'n_occ': len(occ),
            'n_pages': len(pages),
            'num_split': num_split,
            'unit_split': unit_split,
            'variants': [
                {'cn': v[0][5][:300],
                 'count': len(v),
                 'where': sorted({f'{o[0]}:{page_of(o[2])}' for o in v})[:6]}
                for v in (by_num if num_split else by_unit).values()
            ],
        })

    findings.sort(key=lambda f: (-f['n_pages'], -f['n_occ']))
    print(f'对齐成功叶子 {aligned}，因块数不等跳过 {skipped}')
    print(f'含数字且长度>=24 的英文块（去重后）{len(groups)} 组，其中重复出现的 '
          f'{sum(1 for v in groups.values() if len(v) > 1)} 组')
    print(f'中文出现数值/单位分歧且跨页 的：**{len(findings)}** 组')
    for f in findings[:40]:
        print('\n' + '=' * 90)
        print('EN :', f['en'][:220])
        print(f"   occ={f['n_occ']} pages={f['n_pages']} num_split={f['num_split']} unit_split={f['unit_split']}")
        for v in f['variants']:
            print(f"   CN x{v['count']}: {v['cn'][:200]}")
            print(f"        @ {v['where']}")
    json.dump(findings, open(os.path.join(OUT, 'p1_crosspage_block.json'), 'w',
                             encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', os.path.join(OUT, 'p1_crosspage_block.json'))


if __name__ == '__main__':
    main()
