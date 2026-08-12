#!/usr/bin/env python3
"""把 `lang/*.json` 拍平成 Foundry 真正查得到的**扁平点号键**，并按英文侧逐键核对。

  python flatten_lang.py --repo <repo> --english <英文 lang 文件> [--write]

踩过的坑（这脚本就是为它写的）
------------------------------
Foundry 查一个翻译键的方式（`foundry.utils.getProperty`）是：

    1. 先看**整个键**在不在对象里：`translations["EMBER.CALENDAR.WORLD"]`
    2. 不在，再按点拆分逐级下探：`translations["EMBER"]["CALENDAR"]["WORLD"]`

我们的 `ember_cn/lang/cn.json` 写成了**两者都不满足的混合形态**：

    "EMBER.CALENDAR": { "WORLD": "世界地图", ... }      ← 顶层键带点，值却是嵌套对象

整键 `EMBER.CALENDAR.WORLD` 不存在；拆分后第一段 `EMBER` 也不存在（只有 `EMBER.CALENDAR`）。
于是**两条路都断**，Foundry 回落到 Ember 官方的英文。实测 486 个键里只有 114 个真正生效，
**372 个（77%）是死的** —— 而且日历那排 tooltip 正好全在里面，用户实测报上来的就是它。

更要命的是这个缺陷**能骗过校验**：早先那个覆盖率脚本自己会递归展开嵌套，
于是报「486 键缺口 0」。**校验必须复刻被验证系统的查找语义，不能用自己的一套。**
本脚本的 `foundry_lookup()` 就是照 `getProperty` 复刻的。
"""
from __future__ import annotations
import argparse
import json
import os


def foundry_lookup(tr, key):
    """复刻 `foundry.utils.getProperty`：先整键命中，否则按点逐级下探。"""
    if key in tr:
        v = tr[key]
        return v if isinstance(v, str) else None
    node = tr
    for p in key.split('.'):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node if isinstance(node, str) else None


def flatten(obj, prefix=''):
    """嵌套对象 -> 扁平点号键。已经是点号键的原样保留。"""
    out = {}
    for k, v in obj.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--english', required=True, help='对照用的英文 lang 文件')
    ap.add_argument('--lang', default='cn')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    cn_path = os.path.join(a.repo, 'lang', f'{a.lang}.json')
    cn = json.load(open(cn_path, encoding='utf-8'))
    en = json.load(open(a.english, encoding='utf-8'))
    en_keys = list(flatten(en))

    before = sum(1 for k in en_keys if foundry_lookup(cn, k))
    flat = flatten(cn)
    after = sum(1 for k in en_keys if foundry_lookup(flat, k))

    print(f'{cn_path}')
    print(f'  顶层键 {len(cn)} -> 拍平后 {len(flat)}')
    print(f'  英文侧共 {len(en_keys)} 个键')
    print(f'  Foundry 查得到的中文：拍平前 **{before}** -> 拍平后 **{after}**')
    still = [k for k in en_keys if not foundry_lookup(flat, k)]
    print(f'  仍查不到 {len(still)}（这些是真的没译）: {still[:6]}')
    extra = [k for k in flat if k not in set(en_keys)]
    if extra:
        print(f'  中文有、英文没有的键 {len(extra)}（上游删过或我们多写了）: {extra[:5]}')

    if a.write:
        json.dump(flat, open(cn_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('  已写回（扁平点号键）')
    else:
        print('  (未加 --write，未改动)')


if __name__ == '__main__':
    main()
