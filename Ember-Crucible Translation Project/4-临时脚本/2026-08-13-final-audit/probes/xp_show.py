#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按英文/中文正则在全库找叶子或块，打印 EN/CN 原文与完整路径。只读。

  python xp_show.py "<en 正则>" [--blocks] [--cn "<cn 正则>"] [--limit N] [--raw]
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import plain, split_blocks, load_all


def main():
    args = sys.argv[1:]
    pat = re.compile(args[0], re.I) if args and not args[0].startswith('--') else None
    blocks = '--blocks' in args
    raw = '--raw' in args
    limit = 40
    cnpat = None
    for i, a in enumerate(args):
        if a == '--limit':
            limit = int(args[i + 1])
        if a == '--cn':
            cnpat = re.compile(args[i + 1])
    n = 0
    for repo, pack, path, en, cn in load_all():
        cn = cn or ''
        if blocks:
            eb, cb = split_blocks(en), split_blocks(cn)
            if len(eb) != len(cb):
                continue
            for j, (e, c) in enumerate(zip(eb, cb)):
                pe, pc = plain(e), plain(c)
                ok = (pat.search(pe) if pat else True) and (cnpat.search(pc) if cnpat else True)
                if pe and ok:
                    n += 1
                    if n > limit:
                        return
                    print(f'\n--- {repo} / {pack} / {path} [block {j}]')
                    print('EN:', pe[:700])
                    print('CN:', pc[:700])
        else:
            ok = (pat.search(en) if pat else True) and (cnpat.search(cn) if cnpat else True)
            if ok:
                n += 1
                if n > limit:
                    return
                print(f'\n--- {repo} / {pack} / {path}')
                if raw:
                    print('EN:', en)
                    print('CN:', cn)
                else:
                    print('EN:', plain(en)[:1000])
                    print('CN:', plain(cn)[:1000])


if __name__ == '__main__':
    main()
