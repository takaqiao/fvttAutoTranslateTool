#!/usr/bin/env python3
"""对每条 drift 条目判「中文跟的是旧英文还是新英文」——用**标记指纹**，不看长度。

长度比值（scan_en_drift 的 stale 判据）是启发式，实测假阳性极多。
真正能定案的是「只出现在旧英文 / 只出现在新英文」的那些**会被原样抄进译文**的记号：
`@UUID[...]`、`[[/...]]`、`class="..."`、`@Advantage[n]`、`@CriticalSuccess[n]`、
`data-system=`、`<h4>` 标题数、`<li>` 个数……

输出每条：
  OLD_ONLY_HIT  中文里出现了**只有旧英文才有**的记号  → 强 STALE 信号
  NEW_ONLY_MISS 中文里缺了**只有新英文才有**的记号    → 强 STALE / PARTIAL 信号
  两者都空                                            → 大概率 OK（仍需抽读散文）
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

PAT = re.compile(
    r'@UUID\[[^\]]*\]'
    r'|\[\[/[^\]]*\]\]'
    r'|@Advantage\[[^\]]*\]|@Disadvantage\[[^\]]*\]'
    r'|@CriticalSuccess\[[^\]]*\]|@CriticalFailure\[[^\]]*\]'
    r'|@Embed\[[^\]]*\]|@Check\[[^\]]*\]'
    r'|class="[^"]*"'
    r'|data-system="[^"]*"'
    r'|<(?:/?)(?:p|li|ul|ol|h1|h2|h3|h4|h5|h6|section|div|table|tr|td|th|blockquote|figure|figcaption|sup|sub|em|strong)\b'
)


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
    return Counter(PAT.findall(s or ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=25)
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
        so, sn, sc = sig(o.get(p)), sig(n.get(p)), sig(c.get(p))
        # 只看「旧英与新英计数不同」的记号：中文的计数偏向哪一边，就是跟了哪一版
        keys = [k for k in set(so) | set(sn) if so.get(k, 0) != sn.get(k, 0)]
        like_old, like_new, neither = [], [], []
        for k in sorted(keys):
            a_, b_, c_ = so.get(k, 0), sn.get(k, 0), sc.get(k, 0)
            row = f'{k} 旧{a_}/新{b_}/中{c_}'
            if c_ == b_:
                like_new.append(row)
            elif c_ == a_:
                like_old.append(row)
            else:
                neither.append(row)
        verdict = 'OK?' if not like_old and not neither else 'CHECK'
        print(f'[{i}] {verdict} {p}')
        if like_old:
            print('    ★跟旧英文: ' + ' | '.join(like_old))
        if neither:
            print('    ?两边都不等: ' + ' | '.join(neither))
        if like_new and (like_old or neither):
            print('    (跟新英文: ' + ' | '.join(like_new) + ')')


if __name__ == '__main__':
    main()
