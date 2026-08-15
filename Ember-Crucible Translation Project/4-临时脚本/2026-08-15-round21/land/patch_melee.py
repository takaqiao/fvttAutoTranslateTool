#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十一轮落盘：14 条可疑叶里唯一一条**真的跟着旧英文**的地方。

叶：`2-Crucible汉化插件/compendium/cn/crucible.rules.json`
     :: Combat.pages.Engagement and Flanking.text
旧英文：… applies the correct number of boons to **attack rolls** against the Flanked creature.
新英文：… applies the correct number of boons to **melee attack rolls** against the Flanked creature.
中文  ：… 施加正确数量的恩惠骰（**缺「近战」**）
同叶上文那句 “melee attack rolls” 中文已作「近战攻击检定」，所以缺的就是这两个字，不是译法分歧。

⚠ 反空转 / 反顺序覆盖：
  · 只按**唯一整串**定位（全库实测该串 1 处），命中数 != 1 直接非零退出，不猜不模糊匹配。
  · 只改这一处，改完逐叶 diff 整份包，**除本叶外任何叶变了都算失败**。
  · 默认 --dry；加 --write 才落盘。
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PACK = os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn', 'crucible.rules.json')
PATH = ['Combat', 'pages', 'Engagement and Flanking', 'text']

OLD = '夹击效果会自动为针对该被夹击生物的攻击检定施加正确数量的恩惠骰。'
NEW = '夹击效果会自动为针对该被夹击生物的近战攻击检定施加正确数量的恩惠骰。'


def leaves(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, p + '/' + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, p + '/%d' % i)
    elif isinstance(o, str):
        yield p, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    raw = open(PACK, encoding='utf-8').read()
    before = json.loads(raw)

    # ① 全包命中计数 —— 必须恰好 1
    hits = [p for p, s in leaves(before) if OLD in s]
    print(f'全包字符串叶 {sum(1 for _ in leaves(before))} 条 · 命中旧串的叶 {len(hits)} 条 -> {hits}')
    if len(hits) != 1:
        print('** 命中数不是 1，拒绝改动 **')
        return 2

    cur = before['entries']
    for seg in PATH[:-1]:
        cur = cur[seg]
    val = cur[PATH[-1]]
    if OLD not in val:
        print('** 目标叶里没有旧串，拒绝改动 **')
        return 2
    if val.count(OLD) != 1:
        print(f'** 目标叶里旧串出现 {val.count(OLD)} 次，拒绝改动 **')
        return 2

    new_val = val.replace(OLD, NEW)
    print(f'  叶长 {len(val)} -> {len(new_val)}（+{len(new_val) - len(val)}）')
    print(f'  OLD: …{OLD}')
    print(f'  NEW: …{NEW}')

    if not a.write:
        print('(--dry：未落盘)')
        return 0

    cur[PATH[-1]] = new_val
    with open(PACK, 'w', encoding='utf-8') as f:
        json.dump(before, f, ensure_ascii=False, indent=2)
        f.write('\n')

    # ② 落后逐叶回读：除本叶外一条都不许变
    after = json.load(open(PACK, encoding='utf-8'))
    b = dict(leaves(json.loads(raw)))
    c = dict(leaves(after))
    changed = [k for k in set(b) | set(c) if b.get(k) != c.get(k)]
    print(f'  逐叶复核：变动叶 {len(changed)} 条 -> {changed}')
    if len(changed) != 1 or not changed[0].endswith('/Engagement and Flanking/text'):
        print('** 变动面超出本叶，请立即回滚 **')
        return 3
    print('  ✅ 已落盘，且只动了这一叶')
    return 0


if __name__ == '__main__':
    sys.exit(main())
