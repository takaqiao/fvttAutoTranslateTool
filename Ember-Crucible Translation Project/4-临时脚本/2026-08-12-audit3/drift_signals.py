#!/usr/bin/env python3
"""对 drift 的一段区间做**机械信号**普查，替人先把「肯定 stale」的挑出来。

判据不是长度比值，而是「新英文里有、中文里没有」的**可照抄标记**：
标记本身不翻译，所以中文里缺了它，就是中文没跟上新英文的硬证据。
反之「旧英文有、新英文没有、中文却有」＝中文还留着上游删掉的东西。

信号：
  * class="complex-check" / class="advantage" / class="critical-success" 等功能性 class
  * [[/...]] 内联命令体（整段照抄）
  * @UUID[...] 方括号内的目标
  * @Advantage[...] / @CriticalSuccess[...] 的参数
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

PATTERNS = {
    'class': re.compile(r'class="([^"]+)"'),
    'inline': re.compile(r'\[\[/[^\]]+\]\]'),
    'uuid': re.compile(r'@UUID\[([^\]]+)\]'),
    'enrich': re.compile(r'@(?:Advantage|Disadvantage|CriticalSuccess|CriticalFailure|Embed|Check|Condition|Action)\[[^\]]*\]'),
    'datasys': re.compile(r'data-system="[^"]+"'),
}


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


def sig(s):
    out = {}
    for k, rx in PATTERNS.items():
        out[k] = Counter(m.group(0) for m in rx.finditer(s or ''))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=51)
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'] if a.bucket == 'stale' else d['all_changed_with_cn']
    items = items[a.start:a.start + a.limit]

    old_map = baseline_packs(a.baseline)
    cache = {}

    def pack_leaves(pack):
        if pack in cache:
            return cache[pack]
        o, n, c = {}, {}, {}
        if pack in old_map:
            leaves(load_json(old_map[pack]).get('entries', {}), [], o)
        for sub, box in (('en', n), ('cn', c)):
            p = os.path.join(a.repo, 'compendium', sub, pack)
            if os.path.exists(p):
                leaves(load_json(p).get('entries', {}), [], box)
        cache[pack] = (o, n, c)
        return cache[pack]

    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        so, sn, sc = sig(o.get(p)), sig(n.get(p)), sig(c.get(p))
        rows = []
        for k in PATTERNS:
            # 新英文有、中文没有（数量少了）
            miss = sn[k] - sc[k]
            # 中文有、新英文没有（多半是旧英文的残留）
            extra = sc[k] - sn[k]
            # 只报「与旧英文有关」的：旧英文有而新英文没有 → 残留；新有旧无 → 未跟上
            for tokv, cnt in miss.items():
                tagv = 'NEW-ONLY' if so[k][tokv] < sn[k][tokv] else 'missing'
                rows.append(f'  CN缺 {k}[{tagv}] x{cnt}: {tokv[:120]}')
            for tokv, cnt in extra.items():
                tagv = 'OLD-RESIDUE' if so[k][tokv] > sn[k][tokv] else 'extra'
                rows.append(f'  CN多 {k}[{tagv}] x{cnt}: {tokv[:120]}')
        flag = 'FLAG' if any('NEW-ONLY' in r or 'OLD-RESIDUE' in r for r in rows) else ('minor' if rows else 'clean')
        print(f'[{i}] {flag} {p}')
        for r in rows:
            print(r)


if __name__ == '__main__':
    main()
