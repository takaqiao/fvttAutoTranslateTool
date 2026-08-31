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
    """遍历到叶子。**必须下钻 list** —— 索引作为一段路径，与 `apply_translations.py`
    的 `split_path` / `get_at` 同一套语义。

    2026-08-12 之前只认 dict 与 str，于是数组内的叶子在两侧**都**看不见：
    英文键集里没有、中文键集里也没有，两边一起瞎，所以本脚本一路报「死键 0」，
    而 PROJECT.md 第 7 节 F 项点名说已清掉的 `items.吞噬思维 Devour Thoughts`
    （把译名当成了键）原样还在（审计 2026-08-12 第 2.1 条）。
    """
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f'{p}.{k}' if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f'{p}.{i}' if p else str(i))
    elif isinstance(o, str):
        yield p


def prune(node, dead, path=''):
    """按点号路径删除，并清掉因此变空的父节点。返回删除数。

    ⚠ **list 只做尾部截断。** 数组中间的元素一旦删掉，它后面每个元素的索引都会前移
    一位 —— 也就是把一批**正在生效**的译文整体挪到别人的键上去，比留着死键坏得多。
    所以中间的死叶子只统计、不删；`main()` 会把「报出来的」与「真删掉的」分开打印，
    差额就是这类。
    """
    n = 0
    if isinstance(node, list):
        for i in range(len(node) - 1, -1, -1):
            sub = f'{path}.{i}' if path else str(i)
            v = node[i]
            tail = (i == len(node) - 1)
            if isinstance(v, (dict, list)):
                n += prune(v, dead, sub)
                if not v and tail:
                    del node[i]
            elif sub in dead and tail:
                del node[i]
                n += 1
        return n
    for k in list(node):
        sub = f'{path}.{k}' if path else k
        v = node[k]
        if isinstance(v, (dict, list)):
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
    ap.add_argument('--show', type=int, default=0,
                    help='每个包打印前 N 条死键（**加 --write 之前先用它逐条核对判据本身**：'
                         '2026-08-12 抽取器按 _id 建键那次，300 条正在生效的地图针脚译名'
                         '被判成死键，prune_dead 差一步就整批删掉）')
    ap.add_argument('--out', help='把死键清单写成 JSON，便于逐条核对')
    a = ap.parse_args()

    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    en_dir = os.path.join(a.repo, 'compendium', 'en')
    total = total_bytes = cjk_keys = removed = 0
    listing = {}
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
        listing[fn] = sorted(dead)
        if a.write:
            removed += prune(cn_doc, dead)
            with open(cnp, 'w', encoding='utf-8') as f:
                json.dump(cn_doc, f, ensure_ascii=False, indent=2)
            total_bytes += before - os.path.getsize(cnp)
        total += len(dead)
        print(f'  {fn:<44} 死键 {len(dead):>4}' + (f'   ⚠ 键名含中文 {len(bad)}' if bad else ''))
        for k in bad:
            print(f'      ⚠ {k}')
        for k in sorted(dead)[:a.show]:
            if k not in bad:
                print(f'        {k}')

    print(f'\n合计死键 {total}（键名含中文 {cjk_keys}）'
          + (f'，实删 {removed}，省下 {total_bytes/1024:.0f} KB' if a.write
             else '（未加 --write，未改动）'))
    if a.write and removed < total:
        print(f'  注：{total - removed} 条在数组中间，删掉会让后续元素索引前移，故保留（见 prune 的说明）')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(listing, f, ensure_ascii=False, indent=1)
        print(f'  → {a.out}')


if __name__ == '__main__':
    main()
