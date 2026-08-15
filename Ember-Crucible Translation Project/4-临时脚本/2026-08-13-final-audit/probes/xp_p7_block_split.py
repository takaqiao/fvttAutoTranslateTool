#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7：**块级**同英文多中文（scan_same_en_split 的子叶版）。

scan_same_en_split 以「整叶英文全串」为键；本探针以**单块**为键，
于是「同一条规则被抄进不同 journal 的某一段」这一层第一次可见。
默认只看**含数字**的块（本镜头是数值/规则一致性），`--all` 可看全部。

输出按「中文变体数」和「跨的 page 数」排序，去掉纯空白差异。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import CJK, plain, split_blocks, load_all

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)
NUM = re.compile(r'\d')


def page_of(path):
    p = path.split('.')
    return f'{p[1]}/{p[3]}' if len(p) >= 4 and p[2] == 'pages' else '.'.join(p[:2])


def nows(s):
    return re.sub(r'[\s　]+', '', s)


def main():
    want_all = '--all' in sys.argv
    minlen = 24
    for i, a in enumerate(sys.argv):
        if a == '--minlen':
            minlen = int(sys.argv[i + 1])
    groups = defaultdict(list)
    for repo, pack, path, en, cn in load_all():
        if not cn or not CJK.search(cn):
            continue
        eb, cb = split_blocks(en), split_blocks(cn)
        if len(eb) != len(cb):
            continue
        for i, (e, c) in enumerate(zip(eb, cb)):
            pe, pc = plain(e), plain(c)
            if len(pe) < minlen or not CJK.search(pc):
                continue
            if not want_all and not NUM.search(pe):
                continue
            groups[re.sub(r'\s+', ' ', pe).strip().lower()].append(
                (repo, pack, path, i, pc))
    rows = []
    for key, occ in groups.items():
        variants = defaultdict(list)
        for o in occ:
            variants[nows(o[4])].append(o)
        if len(variants) < 2:
            continue
        pages = {page_of(o[2]) for o in occ}
        if len(pages) < 2:
            continue
        rows.append({
            'en': key[:500], 'n_occ': len(occ), 'n_pages': len(pages),
            'n_variants': len(variants),
            'variants': [{'cn': v[0][4][:300], 'n': len(v),
                          'where': sorted({f'{o[0]}:{page_of(o[2])}' for o in v})[:5]}
                         for v in sorted(variants.values(), key=lambda v: -len(v))],
        })
    rows.sort(key=lambda r: (-r['n_variants'], -r['n_pages']))
    print(f'重复出现的英文块组：{sum(1 for v in groups.values() if len(v) > 1)}')
    print(f'其中中文有 ≥2 种写法且跨 ≥2 个 page 的：**{len(rows)}**')
    for r in rows[:70]:
        print('\n' + '=' * 92)
        print(f'[{r["n_variants"]} 种 / {r["n_occ"]} 处 / {r["n_pages"]} 页] EN: {r["en"][:230]}')
        for v in r['variants']:
            print(f'   x{v["n"]}: {v["cn"][:230]}')
            print(f'        @ {v["where"]}')
    json.dump(rows, open(os.path.join(OUT, 'xp_p7_block_split.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', os.path.join(OUT, 'xp_p7_block_split.json'))


if __name__ == '__main__':
    main()
