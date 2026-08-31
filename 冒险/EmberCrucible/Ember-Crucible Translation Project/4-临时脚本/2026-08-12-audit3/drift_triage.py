#!/usr/bin/env python3
"""drift 条目的机械分诊：**中文跟的是旧英文还是新英文**。

依据：`@UUID[...]`、`[[/... ]]`、`data-anchor="..."`、`class="..."` 这类标记在译文里是
**逐字节照抄**的，不会被翻译。所以拿三方的标记多重集就能得到硬信号：

  * CN 含「只在旧英文出现」的标记  → 中文还照着旧版写（STALE 证据）
  * CN 缺「只在新英文出现」的标记  → 新增内容没跟上（STALE/PARTIAL 证据）
  * CN 含新标记且不含旧标记        → 早就照新英文重译过了（OK 证据）

再加一个块数三方对比（<p>/<li>/<h*>/<section> 起头的块）。
标记全等的条目仍可能有纯散文改写，需要人读；本脚本只负责把「有硬信号」的挑出来先看。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

TOKEN = re.compile(
    r'@UUID\[[^\]]*\]'
    r'|@Embed\[[^\]]*\]'
    r'|\[\[/[^\]]*\]\]'
    r'|data-anchor="[^"]*"'
    r'|data-system="[^"]*"'
    r'|class="[^"]*"'
)
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


def toks(s):
    return Counter(TOKEN.findall(s or ''))


def nblocks(s):
    return len([p for p in BLOCK.split(s or '') if p.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--show', type=int, default=6, help='每类最多列几个标记')
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
        pe = os.path.join(a.repo, 'compendium', 'en', pack)
        pc = os.path.join(a.repo, 'compendium', 'cn', pack)
        if os.path.exists(pe):
            leaves(load_json(pe).get('entries', {}), [], n)
        if os.path.exists(pc):
            leaves(load_json(pc).get('entries', {}), [], c)
        cache[pack] = (o, n, c)
        return cache[pack]

    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        oe, ne, cn = o.get(p), n.get(p), c.get(p)
        to, tn, tc = toks(oe), toks(ne), toks(cn)
        only_old = to - tn          # 旧有新无
        only_new = tn - to          # 新有旧无
        cn_has_old = only_old & tc  # 中文里还留着旧标记
        cn_has_new = only_new & tc  # 中文里已有新标记
        cn_miss_new = only_new - tc  # 新标记中文没有
        verdict = []
        if cn_has_old:
            verdict.append(f'留旧×{sum(cn_has_old.values())}')
        if cn_has_new:
            verdict.append(f'有新×{sum(cn_has_new.values())}')
        if cn_miss_new:
            verdict.append(f'缺新×{sum(cn_miss_new.values())}')
        if not only_old and not only_new:
            verdict.append('标记无变化(纯散文改)')
        print(f'\n[{i}] {p}')
        print(f'    blocks 旧{nblocks(oe)} 新{nblocks(ne)} 中{nblocks(cn)} | '
              f'EN {it["en_len_old"]}->{it["en_len_new"]} CN {it["cn_len"]} | ' + ' '.join(verdict))
        if cn_has_old:
            print('    留旧: ' + ' ; '.join(list(cn_has_old.elements())[:a.show]))
        if cn_miss_new:
            print('    缺新: ' + ' ; '.join(list(cn_miss_new.elements())[:a.show]))
        if cn_has_new:
            print('    有新: ' + ' ; '.join(list(cn_has_new.elements())[:a.show]))


if __name__ == '__main__':
    main()
