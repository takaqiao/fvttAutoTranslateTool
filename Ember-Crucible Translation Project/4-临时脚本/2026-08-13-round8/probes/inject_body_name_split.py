#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scan_body_name_splits.py 的**灵敏度回测**：往临时副本里注入已知的正文专名分裂。

绝不碰真库：先把 `--repo` 的 compendium/ 整棵复制到 `--dest`，只改副本。

  python inject_body_name_split.py --repo "2-Crucible汉化插件" --dest <tmp> \
         --swap "扎拉贾=扎拉迦:2" --swap "反制法术=破法术:3"

`--swap 原译名=注入译名:改几片正文叶` —— 只改**非 name** 叶，模拟「name 字段没动、
正文里冒出第二套译名」这一真实故障形态。
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def mutate(node, key, old, new, budget, path=''):
    """就地替换字符串叶里的 old->new，跳过 name/tokenName 叶。返回剩余预算。"""
    if budget[0] <= 0:
        return
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, str):
                p = f'{path}.{k}'
                if k in ('name', 'tokenName') or old not in v:
                    continue
                node[k] = v.replace(old, new)
                budget[0] -= 1
                print(f'   注入 {old}->{new} @ {p}')
                if budget[0] <= 0:
                    return
            else:
                mutate(v, k, old, new, budget, f'{path}.{k}')
                if budget[0] <= 0:
                    return
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                continue
            mutate(v, i, old, new, budget, f'{path}.{i}')
            if budget[0] <= 0:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--swap', action='append', required=True,
                    help='原译名=注入译名:片数')
    a = ap.parse_args()

    dest = os.path.abspath(a.dest)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    shutil.copytree(os.path.join(a.repo, 'compendium'),
                    os.path.join(dest, 'compendium'))
    print(f'副本 -> {dest}')

    cn_dir = os.path.join(dest, 'compendium', 'cn')
    for spec in a.swap:
        pair, _, k = spec.partition(':')
        old, _, new = pair.partition('=')
        budget = [int(k or 2)]
        print(f'-- swap {old} -> {new} ×{budget[0]}')
        for pack in sorted(os.listdir(cn_dir)):
            if not pack.endswith('.json') or budget[0] <= 0:
                continue
            fp = os.path.join(cn_dir, pack)
            with open(fp, encoding='utf-8-sig') as f:
                data = json.load(f)
            before = budget[0]
            mutate(data.get('entries', {}), None, old, new, budget)
            if budget[0] != before:
                with open(fp, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=1)
        if budget[0] > 0:
            print(f'   !! 预算没用完，只注入了部分（剩 {budget[0]}）')


if __name__ == '__main__':
    main()
