#!/usr/bin/env python3
"""对 drift 报告里的每条，打印 **旧英文 → 新英文** 的**块级** diff（只出变更块）。

`drift_dump.py` 打的是三段全文，长条目一条就吃掉几千 token；但 stale 桶里绝大多数条目
的 delta 只有几十字符，真正需要看的只是「上游到底改了哪几句」。

词级 diff 在「整句重写」处会错位（difflib 把不相关的词配成对），所以这里按
**HTML 块边界**（`</p> </li> </h3> </blockquote>` 等）切段再 diff：
段与段之间用相似度配对，改写的段成对打印旧/新全文，纯增/纯删单独标出。

用法：
  python en_diff.py --drift <drift_*.json> --repo <repo> --baseline <旧基准目录>
                    [--bucket stale] [--start N] [--limit N]
                    [--cn] [--cn-grep <正则>]
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 块结束标签：在它之后切开
BLOCK_END = re.compile(
    r'(</(?:p|li|h1|h2|h3|h4|h5|h6|blockquote|tr|td|th|section|ul|ol|figure|figcaption|div|table|tbody|thead)>)',
    re.I)


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


def blocks(s):
    """按块结束标签切段；没有标签的纯文本整体作一段。"""
    if not s:
        return []
    parts = BLOCK_END.split(s)
    out, buf = [], ''
    for p in parts:
        buf += p
        if BLOCK_END.fullmatch(p or ''):
            if buf.strip():
                out.append(buf)
            buf = ''
    if buf.strip():
        out.append(buf)
    return out or [s]


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def show_diff(old, new, sim=0.55):
    a, b = [norm(x) for x in blocks(old)], [norm(x) for x in blocks(new)]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != 'equal']
    if not ops:
        print('  （块级完全相同 —— 只有块内空白差异）')
        return 0
    n = 0
    for tag, i1, i2, j1, j2 in ops:
        oldb, newb = a[i1:i2], b[j1:j2]
        # 在 replace 块内再按相似度配对，避免整段整段地打
        if tag == 'replace' and len(oldb) == len(newb):
            for x, y in zip(oldb, newb):
                n += 1
                if difflib.SequenceMatcher(None, x, y).ratio() >= 0.9:
                    print(f'  ~小改 旧| {x}')
                    print(f'       新| {y}')
                else:
                    print(f'  ~改写 旧| {x}')
                    print(f'       新| {y}')
        else:
            for x in oldb:
                n += 1
                print(f'  -删/旧| {x}')
            for y in newb:
                n += 1
                print(f'  +增/新| {y}')
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale', choices=['stale', 'changed'])
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--cn', action='store_true', help='一并打印现有中文全文')
    ap.add_argument('--cn-grep', default=None, help='只打印中文里匹配该正则的片段（含前后 150 字）')
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'] if a.bucket == 'stale' else d['all_changed_with_cn']
    sel = items[a.start:a.start + a.limit]

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

    for i, it in enumerate(sel, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        print('=' * 90)
        print(f'[{i}] {p}')
        print(f'    EN {it["en_len_old"]}→{it["en_len_new"]} (Δ{it["delta"]}) | CN {it["cn_len"]}')
        show_diff(o.get(p), n.get(p))
        if a.cn:
            print('--- 中文 ---')
            print(c.get(p))
        elif a.cn_grep:
            cn = c.get(p) or ''
            for m in re.finditer(a.cn_grep, cn):
                s = max(0, m.start() - 150)
                print(f'  CN…{cn[s:m.end() + 150]}…')


if __name__ == '__main__':
    main()
