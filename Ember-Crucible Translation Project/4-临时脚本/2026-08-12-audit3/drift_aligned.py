#!/usr/bin/env python3
"""按块位置把 **中文** 对齐到 **新英文**，只打印上游改动过的那些块的三元组。

背景：全库 BLOCK 漂移已是 0（`scan_markup_drift.py`），也就是说每条中文的块签名
与**当前**英文相等 —— 所以「中文块数 == 新英文块数」是既成事实，不含任何信息量，
`stale` 的长度比判据同样被 `@UUID[...]` 密度污染（标记逐字节照抄，会抬高中文/英文比）。

真正要问的是：**上游改过的那几个块，中文跟的是旧的还是新的。**
本脚本先用 difflib 在旧英文块序列与新英文块序列之间求 opcode，
再把 CN 的第 i 块（位置对齐）贴到新英文第 i 块旁边，只输出 replace/insert 段。
delete 段（上游删掉的块）单独列出，用来确认中文里没有它们的残留。

用法：
  python drift_aligned.py --drift <json> --repo <repo> --baseline <dir>
      [--start N] [--limit N] [--max-block N] [--only-misaligned]
"""
from __future__ import annotations
import argparse, json, os, re, sys, difflib

sys.stdout.reconfigure(encoding='utf-8')

BLOCK_RE = re.compile(r'<(p|li|h[1-6]|td|th|figcaption|blockquote)\b[^>]*>(.*?)</\1>', re.S | re.I)
TAG = re.compile(r'<[^>]+>')


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
        out.append(txt)
    if not out:
        out.append(re.sub(r'\s+', ' ', TAG.sub('', html)).strip())
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
    ap.add_argument('--max-block', type=int, default=260)
    ap.add_argument('--summary', action='store_true', help='只打一行汇总')
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

    def clip(s, w=None):
        w = w or a.max_block
        return s if len(s) <= w else s[:w] + ' …'

    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        oe, ne, cn = o.get(p) or '', n.get(p) or '', c.get(p) or ''
        ob, nb, cb = blocks(oe), blocks(ne), blocks(cn)
        aligned = (len(nb) == len(cb))
        print('\n' + '=' * 96)
        print(f'[{i}] {p.split(".journals.")[-1]}')
        print(f'    块 旧EN {len(ob)} | 新EN {len(nb)} | CN {len(cb)}  '
              f'{"对齐" if aligned else "!! 块数不等，位置对齐不可用 !!"}  '
              f'chars {it["en_len_old"]}→{it["en_len_new"]} cn {it["cn_len"]}')
        if a.summary:
            continue

        sm = difflib.SequenceMatcher(None, [norm(x) for x in ob], [norm(x) for x in nb])
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            if tag == 'delete':
                for x in ob[i1:i2]:
                    print(f'  [删] OLD  {clip(x)}')
                continue
            for k in range(j1, j2):
                oldtxt = ob[i1 + (k - j1)] if tag == 'replace' and i1 + (k - j1) < i2 else None
                print(f'  --- 块#{k} ({tag})')
                if oldtxt is not None:
                    print(f'    OLD {clip(oldtxt)}')
                print(f'    NEW {clip(nb[k])}')
                print(f'    CN  {clip(cb[k], (a.max_block // 2) or 130) if aligned and k < len(cb) else "(无法对齐)"}')
            if tag == 'replace' and (i2 - i1) > (j2 - j1):
                for x in ob[i1 + (j2 - j1):i2]:
                    print(f'  [删] OLD  {clip(x)}')


if __name__ == '__main__':
    main()
