#!/usr/bin/env python3
"""旧英文 vs 新英文的**块级** diff，外加中文块数与关键锚点比对。

drift_dump.py 是把三段全文都打出来，读 51 条会吃掉整个上下文。
这个脚本只回答两个问题：
  1. 上游到底删了/加了哪些块（按 <p>/<li>/<h*>/<td> 切）；
  2. 中文里有没有「只存在于新英文」的锚点（@UUID 目标、[[/命令]]、数字）——
     有 → 中文已照新英文写过（stale 是假阳性）。

用法：
  python drift_blockdiff.py --drift <json> --repo <repo> --baseline <dir>
      [--start N] [--limit N] [--max-block N]
"""
from __future__ import annotations
import argparse, json, os, re, sys, difflib

sys.stdout.reconfigure(encoding='utf-8')

BLOCK_RE = re.compile(r'<(p|li|h[1-6]|td|th|figcaption|blockquote)\b[^>]*>(.*?)</\1>', re.S | re.I)
TAG = re.compile(r'<[^>]+>')
UUID_TARGET = re.compile(r'@(?:UUID|Embed)\[([^\]]+)\]')
INLINE_CMD = re.compile(r'\[\[/([a-zA-Z]+)\s*([^\]]*)\]\]')


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


def blocks(html):
    if not html:
        return []
    out = []
    for m in BLOCK_RE.finditer(html):
        txt = TAG.sub('', m.group(2))
        txt = re.sub(r'\s+', ' ', txt).strip()
        if txt:
            out.append(txt)
    if not out:
        txt = re.sub(r'\s+', ' ', TAG.sub('', html)).strip()
        if txt:
            out.append(txt)
    return out


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=51)
    ap.add_argument('--max-block', type=int, default=300)
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'][a.start:a.start + a.limit]
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

    def clip(s):
        return s if len(s) <= a.max_block else s[:a.max_block] + ' …'

    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        oe, ne, cn = o.get(p) or '', n.get(p) or '', c.get(p) or ''
        ob, nb, cb = blocks(oe), blocks(ne), blocks(cn)
        print('\n' + '=' * 96)
        print(f'[{i}] {p}')
        print(f'    EN blocks 旧{len(ob)} → 新{len(nb)} | CN blocks {len(cb)} | '
              f'chars {it["en_len_old"]}→{it["en_len_new"]} cn {it["cn_len"]}')

        sm = difflib.SequenceMatcher(None, [norm(x) for x in ob], [norm(x) for x in nb])
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            if tag in ('delete', 'replace'):
                for x in ob[i1:i2]:
                    print(f'  -OLD  {clip(x)}')
            if tag in ('insert', 'replace'):
                for x in nb[j1:j2]:
                    print(f'  +NEW  {clip(x)}')

        # 锚点：只在新英文里出现的 UUID 目标 / 内联命令
        ot = set(UUID_TARGET.findall(oe)); nt = set(UUID_TARGET.findall(ne))
        ct = set(UUID_TARGET.findall(cn))
        new_only = nt - ot
        gone = ot - nt
        if new_only:
            hit = sorted(x for x in new_only if x in ct)
            miss = sorted(x for x in new_only if x not in ct)
            print(f'  ~ 新增UUID目标 {len(new_only)}: 中文已有 {hit} | 中文缺 {miss}')
        if gone:
            still = sorted(x for x in gone if x in ct)
            if still:
                print(f'  ! 上游已删的UUID目标仍在中文里: {still}')
        oc = set(m.group(0) for m in INLINE_CMD.finditer(oe))
        nc = set(m.group(0) for m in INLINE_CMD.finditer(ne))
        cc = set(m.group(0) for m in INLINE_CMD.finditer(cn))
        nco = nc - oc
        gco = oc - nc
        if nco:
            print(f'  ~ 新增内联命令: 中文已有 {sorted(x for x in nco if x in cc)} | '
                  f'中文缺 {sorted(x for x in nco if x not in cc)}')
        if gco:
            still = sorted(x for x in gco if x in cc)
            if still:
                print(f'  ! 上游已删的内联命令仍在中文里: {still}')


if __name__ == '__main__':
    main()
