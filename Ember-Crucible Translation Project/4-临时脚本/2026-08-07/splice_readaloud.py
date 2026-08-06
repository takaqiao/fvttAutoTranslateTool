#!/usr/bin/env python3
"""Splice the Chinese `readaloud="..."` values from a parked file into the live batch.

第 3 批的译者发现：`apply_translations.py` 的标记闸把整个 `@Embed[Actor.x readaloud="…"]`
当成必须逐字保留的机关，于是那段**要念给玩家听的旁白**根本没法翻 —— 尽管库里已经有 22 处
是翻过的。译者为了满足「0 拒绝」照抄了英文，把译好的中文另存成
`_pending_embed_readaloud.NOT-A-BATCH.json`。

闸门已修（`markup_signature` 现在把 `="…"` 的值抹平再比对）。但那份搁置件是在跨卷术语核对
**之前**写的，直接套用会把核对定下的译名（泰兰 / 契约…）打回去。所以这里只取 readaloud
的值，其余一概以当前 batch 为准。

  python splice_readaloud.py --dir <单元目录> [--write]
"""
from __future__ import annotations
import argparse
import json
import os
import re

READALOUD = re.compile(r'(readaloud=\s*")([^"]*)(")')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--parked', default='_pending_embed_readaloud.NOT-A-BATCH.json')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    cur_p = os.path.join(a.dir, 'batch.json')
    cur = json.load(open(cur_p, encoding='utf-8'))
    park = json.load(open(os.path.join(a.dir, a.parked), encoding='utf-8'))

    spliced = 0
    for k, cv in cur.items():
        pv = park.get(k)
        if not pv:
            continue
        cur_vals = READALOUD.findall(cv)
        park_vals = READALOUD.findall(pv)
        if not cur_vals or len(cur_vals) != len(park_vals):
            continue
        it = iter(park_vals)

        def sub(m):
            nonlocal spliced
            _, new, _ = next(it)
            if new != m.group(2):
                spliced += 1
            return m.group(1) + new + m.group(3)

        cur[k] = READALOUD.sub(sub, cv)
        print(f'{k}\n   readaloud x{len(cur_vals)}')

    print(f'\n换掉 {spliced} 段 readaloud')
    if a.write:
        with open(cur_p, 'w', encoding='utf-8') as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print('已写回 batch.json')
    else:
        print('(未加 --write)')


main()
