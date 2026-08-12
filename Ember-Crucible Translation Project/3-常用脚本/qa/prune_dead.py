#!/usr/bin/env python3
"""删掉中文包里**英文包已经没有的键**（Babele 永远查不到它们）。

  python prune_dead.py --repo <repo> [--write]

Babele 是拿英文原文的键去中文包里查的。上游删掉或改名一个条目之后，中文里那条就
变成了纯粹的死重量：既不显示、也不会被任何扫描发现（覆盖率/残留/签名都只看
「英文有、中文有没有」，反方向不看），只是让每个用户白下几 MB。

顺带能揪出一类真正的缺陷：**键名里混进了中文**。
`items.吞噬思维 Devour Thoughts.name` 这种是把译名当成了键写进去，英文侧的键是
`items.Devour Thoughts`，于是整条翻译对玩家不存在。这类键必然同时出现在死键名单里。
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')


def leaves(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f'{p}.{k}' if p else k)
    elif isinstance(o, str):
        yield p


def prune(node, dead, path=''):
    """按点号路径删除，并清掉因此变空的父节点。返回删除数。"""
    n = 0
    for k in list(node):
        sub = f'{path}.{k}' if path else k
        v = node[k]
        if isinstance(v, dict):
            n += prune(v, dead, sub)
            if not v:
                del node[k]
        elif sub in dead:
            del node[k]
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    en_dir = os.path.join(a.repo, 'compendium', 'en')
    total = total_bytes = cjk_keys = 0
    for fn in sorted(os.listdir(cn_dir)):
        if not fn.endswith('.json'):
            continue
        ep = os.path.join(en_dir, fn)
        if not os.path.exists(ep):
            print(f'  {fn:<44} 没有英文对照，跳过')
            continue
        cnp = os.path.join(cn_dir, fn)
        with open(ep, encoding='utf-8-sig') as f:
            en_keys = set(leaves(json.load(f)))
        with open(cnp, encoding='utf-8-sig') as f:
            cn_doc = json.load(f)
        dead = {k for k in leaves(cn_doc) if k not in en_keys}
        if not dead:
            continue
        bad = [k for k in dead if CJK.search(k)]
        cjk_keys += len(bad)
        before = os.path.getsize(cnp)
        if a.write:
            prune(cn_doc, dead)
            with open(cnp, 'w', encoding='utf-8') as f:
                json.dump(cn_doc, f, ensure_ascii=False, indent=2)
            total_bytes += before - os.path.getsize(cnp)
        total += len(dead)
        print(f'  {fn:<44} 死键 {len(dead):>4}' + (f'   ⚠ 键名含中文 {len(bad)}' if bad else ''))
        for k in bad:
            print(f'      ⚠ {k}')

    print(f'\n合计死键 {total}（键名含中文 {cjk_keys}）'
          + (f'，省下 {total_bytes/1024:.0f} KB' if a.write else '（未加 --write，未改动）'))


if __name__ == '__main__':
    main()
