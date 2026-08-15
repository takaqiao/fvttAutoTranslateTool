#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只印 old→new 的**词级差异**，再把中文全文附上，供逐条判「中文跟的是哪一侧」。

⚠ 反空转：每条印出 old/new 词数与差异块数；差异块数为 0 时显式打「英文没变过」，
   不是静静跳过（那 14 条本就是「英文变过」筛出来的，出现 0 说明取值取错了）。
"""
import difflib
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OLD = os.path.join(P, '5-其他内容', 'english-baseline', 'crucible-cn-0.8.9.1-shipped-en')
NEW = os.path.join(P, '2-Crucible汉化插件', 'compendium', 'en')
CN = os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn')

TAG = re.compile(r'<[^>]+>')


def load(d, fn):
    p = os.path.join(d, fn)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def dig(obj, path):
    cur = obj.get('entries', obj)
    for seg in path.split('.'):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur if isinstance(cur, str) else None


def words(s):
    return re.findall(r'\S+', TAG.sub(' ', s or ''))


def main():
    targets = [l.strip() for l in sys.argv[1:]]
    for t in targets:
        fn, path = t.split('::')
        fn, path = fn.strip(), path.strip()
        o, n, c = load(OLD, fn), load(NEW, fn), load(CN, fn)
        ov, nv, cv = dig(o, path), dig(n, path), dig(c, path)
        ow, nw = words(ov), words(nv)
        print('=' * 96)
        print(f'{fn} :: {path}')
        print(f'  old 词数 {len(ow)} · new 词数 {len(nw)} · cn 字数 {len(cv or "")}')
        blocks = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ow, nw).get_opcodes():
            if tag == 'equal':
                continue
            blocks += 1
            print(f'  [{tag}] OLD: {" ".join(ow[i1:i2])[:400]}')
            print(f'         NEW: {" ".join(nw[j1:j2])[:400]}')
        if blocks == 0:
            print('  ** 英文没变过 —— 取值可能取错了，本条无从判 **')
        print('--- CN 全文 ---')
        print(TAG.sub(' ', cv or ''))


if __name__ == '__main__':
    main()
