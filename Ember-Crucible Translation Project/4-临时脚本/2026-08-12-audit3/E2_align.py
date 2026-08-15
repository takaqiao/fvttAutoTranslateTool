#!/usr/bin/env python3
"""drift 复核的三方**对齐**视图：只打「旧英→新英变了的那几块」以及**位置对应的中文块**。

前提观察（E2 单元实测）：ember 的译文是按块逐块翻的，`stale` 桶里 51 条的
中文块数与**新**英文块数逐条相等。于是块序号可以直接当对齐键 ——
新英文第 j 块的译文就是中文第 j 块。这样复核一条 drift 只需要读三样东西：
被删的旧块、新增/改写的新块、以及新块位置上的中文。整页全文不必读。

块数不等时退化为「打印整段中文」，并在抬头标注 MISALIGNED。
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

BLOCK = re.compile(r'(?=<(?:p|li|h[1-6]|section|div|ul|ol|table|tr|td|th|blockquote|figure|figcaption|aside)\b)', re.I)


def load_json(path):
    raw = open(path, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r',(\s*[}\]])', r'\1', raw))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out['.'.join(path)] = node


def baseline_packs(bdir):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        key = ('ember.crucible-adventure.json' if f == '_repaired.json'
               else f.replace('-en.json', '.json'))
        out.setdefault(key, os.path.join(bdir, f))
    return out


NORM = re.compile(r'[^a-z0-9@\[\]{}#=/]+')


def norm(s):
    """归一化到「中文侧可见的语义」：去大小写、去标点空白。
    保留 @ [ ] { } # = / 因为标记与其参数的差异必须留在视野里。"""
    return NORM.sub('', s.lower())


def split_blocks(s):
    if s is None:
        return []
    parts = [p for p in BLOCK.split(s) if p.strip()]
    return parts if parts else [s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--only', type=int, default=None, help='只看某一条（绝对序号）')
    ap.add_argument('--pad', type=int, default=0, help='变更区两侧多带几块中文')
    ap.add_argument('--maxb', type=int, default=1200, help='单块打印上限')
    ap.add_argument('--skip-cosmetic', action='store_true',
                    help='折叠只差大小写/标点/空白的变更（中文侧看不出来）')
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'] if a.bucket == 'stale' else d['all_changed_with_cn']
    if a.only is not None:
        sel = [(a.only, items[a.only])]
    else:
        sel = list(enumerate(items))[a.start:a.start + a.limit]

    old_map = baseline_packs(a.baseline)
    cache = {}

    def pack_leaves(pack):
        if pack in cache:
            return cache[pack]
        o, n, c = {}, {}, {}
        if pack in old_map:
            leaves(load_json(old_map[pack]).get('entries', {}), [], o)
        pe = os.path.join(a.repo, 'compendium', 'en', pack)
        pc = os.path.join(a.repo, 'compendium', 'cn', pack)
        if os.path.exists(pe):
            leaves(load_json(pe).get('entries', {}), [], n)
        if os.path.exists(pc):
            leaves(load_json(pc).get('entries', {}), [], c)
        cache[pack] = (o, n, c)
        return cache[pack]

    def cut(s):
        return s if len(s) <= a.maxb else s[:a.maxb] + f'…[+{len(s)-a.maxb}]'

    for i, it in sel:
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        oe, ne, cn = o.get(p), n.get(p), c.get(p)
        ob, nb, cb = split_blocks(oe), split_blocks(ne), split_blocks(cn)
        aligned = len(nb) == len(cb)
        print('\n' + '=' * 100)
        print(f'[{i}] {p}')
        print(f'blocks 旧{len(ob)} 新{len(nb)} 中{len(cb)} '
              f'{"对齐" if aligned else "!! MISALIGNED !!"}')
        sm = difflib.SequenceMatcher(None, ob, nb, autojunk=False)
        skipped = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            ot, nt = ''.join(ob[i1:i2]), ''.join(nb[j1:j2])
            if a.skip_cosmetic and norm(ot) == norm(nt):
                skipped += 1
                continue
            sim = round(difflib.SequenceMatcher(None, ot, nt).ratio(), 2)
            print(f'--- {tag} @新块[{j1},{j2}) sim={sim} ---')
            for b in ob[i1:i2]:
                print('OLD- ' + cut(b))
            for b in nb[j1:j2]:
                print('NEW+ ' + cut(b))
            if aligned:
                lo, hi = max(0, j1 - a.pad), min(len(cb), j2 + a.pad)
                if lo == hi:
                    print(f'CN   (新块区间为空，检查 {lo} 附近) ' + cut(cb[lo] if lo < len(cb) else ''))
                for b in cb[lo:hi]:
                    print('CN   ' + cut(b))
        if skipped:
            print(f'(另有 {skipped} 处纯格式差异：大小写/标点/空白，中文侧不可见，已折叠)')
        if not aligned:
            print('--- 中文全文 ---')
            print(cn)


if __name__ == '__main__':
    main()
