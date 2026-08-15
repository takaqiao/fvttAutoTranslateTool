#!/usr/bin/env python3
"""孪生包全库对齐：**英文逐字节相同、中文却不同** 的路径，一次性统一。

ember 有两个包装着同一部战役：
    ember.crucible-adventure.json   crucible 规则侧 —— **主线**（PROJECT.md 第 1 节优先级）
    ember.adventure.json            dnd5e 规则侧 —— 附带项

journal 正文在两包间大量逐字节相同。`qa/propagate_fix.py` 只推「`--since` 之后改过的」，
所以历次审计只覆盖了当轮动过的叶子；**存量的分叉从来没有被系统清过**。
2026-08-13b 实测：两包共有 11359 条路径里，英文相同而中文不同的有 **268 条**，
另有 41 条一侧压根没有中文。

判据与方向
----------
* 只处理**英文逐字节相同**的路径 —— 英文不同的是 dnd5e 与 crucible 的规则文本本来就不一样，
  那是合法分叉（实测 238 条），不能碰。
* 方向取**主线**（crucible 侧）。它才是项目要交付的东西，而且历轮逐句复核读的都是它。
* 一侧没有中文时取有中文的那一侧（无论哪边）。
* 落盘仍走 `apply_translations.py` 的三道闸，所以本脚本只产批次、不写库。

用法：
  python sync_twin_packs.py --repo <repo> [--out-dir <批次目录>] [--show 20]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CJK = re.compile(r'[一-鿿]')
MAIN = 'ember.crucible-adventure.json'
TWIN = 'ember.adventure.json'


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}.{i}')
    elif isinstance(obj, str) and obj:
        yield path, obj


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def role_of(path):
    for seg in reversed(path.split('.')):
        if not seg.isdigit():
            return seg
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out-dir')
    ap.add_argument('--show', type=int, default=20)
    a = ap.parse_args()

    data = {}
    for pack in (MAIN, TWIN):
        en = dict(walk(load(os.path.join(a.repo, 'compendium', 'en', pack)).get('entries', {})))
        cn = dict(walk(load(os.path.join(a.repo, 'compendium', 'cn', pack)).get('entries', {})))
        data[pack] = (en, cn)

    en_m, cn_m = data[MAIN]
    en_t, cn_t = data[TWIN]
    common = set(en_m) & set(en_t)

    to_twin, to_main, stats = {}, {}, Counter()
    for p in sorted(common):
        if en_m[p] != en_t[p]:
            stats['英文不同（合法分叉，不动）'] += 1
            continue
        a_cn, b_cn = cn_m.get(p), cn_t.get(p)
        if a_cn == b_cn:
            stats['已一致'] += 1
            continue
        if a_cn and b_cn:
            stats['**英文同、中文分叉** -> 取主线'] += 1
            to_twin[p] = a_cn
        elif a_cn and not b_cn:
            stats['孪生包缺中文 -> 从主线补'] += 1
            to_twin[p] = a_cn
        elif b_cn and not a_cn:
            stats['主线缺中文 -> 从孪生包补'] += 1
            to_main[p] = b_cn

    print(f'两包共有路径 {len(common)}')
    for k, v in stats.most_common():
        print(f'  {k:34s} {v}')
    print(f'\n要写进孪生包 {len(to_twin)} 条 / 要写进主线 {len(to_main)} 条')

    shown = 0
    for p, v in to_twin.items():
        if shown >= a.show:
            break
        if not CJK.search(v):
            continue
        old = (cn_t.get(p) or '')
        print(f'\n  {role_of(p):16s} {p[-70:]}')
        print(f'    孪生包旧: {old[:100]}')
        print(f'    主线新  : {v[:100]}')
        shown += 1

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        for pack, items in ((TWIN, to_twin), (MAIN, to_main)):
            if not items:
                continue
            f = os.path.join(a.out_dir, f'TWINSYNC__ember__{pack[:-5]}.json')
            with open(f, 'w', encoding='utf-8') as fh:
                json.dump(items, fh, ensure_ascii=False, indent=1)
            print(f'-> {f}  ({len(items)} 条)')


if __name__ == '__main__':
    main()
