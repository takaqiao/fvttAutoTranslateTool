#!/usr/bin/env python3
"""drift_dump.py 的补充：只打印 **旧英→新英的词级 diff**，外加可选的中文全文。

`drift_dump.py` 打三段全文，条目一多就把上下文吃光；而 `stale` 桶的尾部
（delta 只有 -1 ~ -13 字符）实质差异往往只有一个词。
本脚本用 difflib 做词级 opcodes，只输出变动片段及其上下文，
判「上游到底改了什么」快得多。中文默认不打（--cn 打开）。

用法与 drift_dump.py 相同的 --drift/--repo/--baseline/--bucket/--start/--limit。
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

WORD = re.compile(r'\s+|(?<=>)|(?=<)')


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


def tok(s):
    """按空白与标签边界切词，保留原字节（join 后必须等于原串）。"""
    out, buf = [], []
    for ch in s:
        buf.append(ch)
        if ch.isspace():
            out.append(''.join(buf))
            buf = []
    if buf:
        out.append(''.join(buf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale', choices=['stale', 'changed'])
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--ctx', type=int, default=8, help='变动片段两侧保留的词数')
    ap.add_argument('--cn', action='store_true', help='同时打印现有中文全文')
    ap.add_argument('--cn-chars', type=int, default=4000)
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'] if a.bucket == 'stale' else d['all_changed_with_cn']
    total = len(items)
    items = items[a.start:a.start + a.limit]

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

    print(f'# {os.path.basename(a.drift)} bucket={a.bucket} '
          f'共 {total} 条，本次 [{a.start}, {a.start + len(items)})')
    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        old_en, new_en, cn = o.get(p), n.get(p), c.get(p)
        print('\n' + '=' * 96)
        print(f'[{i}] {it["pack"]}')
        print(f'BATCH_PATH: {p}')
        print(f'EN {it["en_len_old"]} -> {it["en_len_new"]} | CN {it["cn_len"]}')
        if old_en is None or new_en is None:
            print('!! 缺旧英文或新英文，跳过 diff')
        else:
            A, B = tok(old_en), tok(new_en)
            sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
            ops = [op for op in sm.get_opcodes() if op[0] != 'equal']
            if not ops:
                print('(词级无差异；差异只在空白)')
            for tag, i1, i2, j1, j2 in ops:
                pre = ''.join(A[max(0, i1 - a.ctx):i1])
                post = ''.join(A[i2:i2 + a.ctx])
                print(f'--- {tag} ---')
                print(f'  ctx-: ...{pre}')
                print(f'  OLD : [[{"".join(A[i1:i2])}]]')
                print(f'  NEW : [[{"".join(B[j1:j2])}]]')
                print(f'  ctx+: {post}...')
        if a.cn:
            print('----- CN -----')
            s = cn or '(缺)'
            print(s if len(s) <= a.cn_chars else s[:a.cn_chars] + f'…[省略{len(s)-a.cn_chars}]')


if __name__ == '__main__':
    main()
