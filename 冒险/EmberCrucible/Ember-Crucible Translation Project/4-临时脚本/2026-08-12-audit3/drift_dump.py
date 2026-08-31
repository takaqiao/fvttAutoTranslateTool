#!/usr/bin/env python3
"""把 scan_en_drift.py 报出的条目，按 **旧英文 / 新英文 / 现有中文** 三元组完整导出。

`scan_en_drift.py --out` 的 JSON 里三段文本都截到 220 字符，只够排序、不够判断。
复核一条 drift 必须同时看全三段：**上游到底改了什么**（旧英→新英的差异），
以及**中文当初是照哪一版写的**。少任何一段都会误判：

  * 只看新英+中文 → 分不清「中文漏译」和「上游后加的内容」；
  * 只看比值      → 阶段 28 已证明它会漏（0.377 的正常比值下藏着凭空增删）。

用法：
  python drift_dump.py --drift <drift_*.json> --repo <repo> --baseline <旧基准目录>
                       [--bucket stale|changed] [--grep <正则，匹配 path>]
                       [--start N] [--limit N] [--max-chars N]

输出是给人（和 agent）读的纯文本；`BATCH_PATH:` 那一行可直接当批次文件的键。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

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
    """与 scan_en_drift.baseline_packs 保持一致：`_repaired.json` 先到先得。"""
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        key = ('ember.crucible-adventure.json' if f == '_repaired.json'
               else f.replace('-en.json', '.json'))
        out.setdefault(key, os.path.join(bdir, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale', choices=['stale', 'changed'])
    ap.add_argument('--grep', default=None, help='正则，匹配 path')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--max-chars', type=int, default=6000,
                    help='单段文本超过此长度就截断（两头各留一半），避免一条吃掉整个上下文')
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'] if a.bucket == 'stale' else d['all_changed_with_cn']
    if a.grep:
        rx = re.compile(a.grep)
        items = [it for it in items if rx.search(it['path'])]
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

    def clip(s):
        if s is None:
            return '(缺)'
        if len(s) <= a.max_chars:
            return s
        h = a.max_chars // 2
        return s[:h] + f'\n…【中间省略 {len(s) - a.max_chars} 字符】…\n' + s[-h:]

    print(f'# {os.path.basename(a.drift)} bucket={a.bucket} '
          f'命中 {total} 条，本次导出 [{a.start}, {a.start + len(items)})')
    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        old_en, new_en, cn = o.get(p), n.get(p), c.get(p)
        print('\n' + '=' * 100)
        print(f'[{i}] PACK: {it["pack"]}')
        print(f'PATH: {p}')
        print(f'BATCH_PATH: {p}')
        print(f'EN 旧 {it["en_len_old"]} → 新 {it["en_len_new"]} 字符 | '
              f'中文 {it["cn_len"]} | 中文/旧={it["ratio_old"]} 中文/新={it["ratio_new"]}')
        print('-' * 40 + ' 旧英文（基准） ' + '-' * 40)
        print(clip(old_en))
        print('-' * 40 + ' 新英文（当前） ' + '-' * 40)
        print(clip(new_en))
        print('-' * 40 + ' 现有中文 ' + '-' * 40)
        print(clip(cn))


if __name__ == '__main__':
    main()
