#!/usr/bin/env python3
"""把「中文没跟上英文」的两类欠账合并切成单元（页文件格式）。

  python prep_catchup.py <每单元权重> --drift <drift.json> [--coverage <cov.json>] --repo <repo>

合并的两类
----------
* **K 类**（`scan_en_drift.py` 的 `stale`）：上游改过英文，中文还停在旧版。
* **M 类**（`scan_content_coverage.py`）：英文里的数字中文没有 —— 这类**英文可能从没变过**，
  是译文自始就漏了具体数值（例：`Swallow` 英文「Size 至少小 2 级」中文只写「比自己更小」）。

两者形状相同：**把中文补到完全覆盖今天的英文**，所以放同一批做，只在 `index.json`
里用 `reason` 标明各自的来由，让 agent 知道该重点看什么。

产出沿用 `prep_realign.py` 的页文件布局，所以 `collect_realign.py` / `diff_realign.py` /
`tagseq.py` / `prose_survival.py` 全部照用。单元**不混包**（一个 batch 只落一个 pack）。
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROOT = os.environ.get("EMBER_PARALLEL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parallel")


def get(root, path):
    n = root
    for k in path.split('.'):
        n = n.get(k) if isinstance(n, dict) else None
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weight', type=int, nargs='?', default=20000)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--drift')
    ap.add_argument('--coverage')
    ap.add_argument('--prefix', default='catchup')
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    cache = {}

    def load(side, pack):
        k = (side, pack)
        if k not in cache:
            p = os.path.join(en_dir if side == 'en' else cn_dir, pack)
            cache[k] = json.load(open(p, encoding='utf-8')).get('entries', {})
        return cache[k]

    merged = {}
    if a.drift:
        d = json.load(open(a.drift, encoding='utf-8'))
        for r in d.get('items', []):
            merged.setdefault((r['pack'], r['path']), []).append(
                f"英文变过（旧 {r['en_len_old']} → 新 {r['en_len_new']} 字符，"
                f"中文/旧={r['ratio_old']} 中文/新={r['ratio_new']}）")
    if a.coverage:
        d = json.load(open(a.coverage, encoding='utf-8'))
        for r in d.get('items', []):
            if not r.get('missing_numbers'):
                continue
            merged.setdefault((r['pack'], r['path']), []).append(
                f"中文缺英文里的数字: {', '.join(r['missing_numbers'][:8])}")

    rows = []
    for (pack, path), reasons in merged.items():
        e, c = get(load('en', pack), path), get(load('cn', pack), path)
        if not isinstance(e, str) or not isinstance(c, str) or not CJK.search(c):
            continue
        rows.append({'pack': pack, 'path': path, 'en': e, 'cn': c,
                     'reason': reasons,
                     'weight': round(0.2 * len(e) + 0.1 * len(c))})

    units = []
    for pack in sorted({r['pack'] for r in rows}):
        pr = sorted([r for r in rows if r['pack'] == pack], key=lambda r: -r['weight'])
        n = max(1, round(sum(r['weight'] for r in pr) / a.weight))
        buckets = [[] for _ in range(n)]
        load_ = [0] * n
        for r in pr:
            i = load_.index(min(load_))
            buckets[i].append(r)
            load_[i] += r['weight']
        units += [b for b in buckets if b]

    manifest = []
    for i, ch in enumerate(units, 1):
        name = f'{a.prefix}-{i}'
        d = os.path.join(ROOT, name)
        pages = os.path.join(d, 'pages')
        os.makedirs(pages, exist_ok=True)
        index = []
        for j, r in enumerate(ch, 1):
            for side in ('en', 'cn'):
                with open(os.path.join(pages, f'{j}.{side}.html'), 'w',
                          encoding='utf-8', newline='') as f:
                    f.write(r[side])
            index.append({'id': j, 'path': r['path'], 'reason': r['reason'],
                          'en_chars': len(TAG.sub('', r['en'])),
                          'cn_chars': len(TAG.sub('', r['cn']))})
        json.dump({'unit': name, 'pack': ch[0]['pack'], 'pages': index},
                  open(os.path.join(d, 'index.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        json.dump({str(r['id']): ch[k]['cn'] for k, r in enumerate(index)},
                  open(os.path.join(d, '_original_cn.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False)
        manifest.append({'unit': name, 'dir': d, 'pack': ch[0]['pack'], 'pages': len(ch)})
        print(f'{name:<13}{len(ch):>4} 条  {ch[0]["pack"]}')

    out = os.path.join(ROOT, f'manifest.{a.prefix}.json')
    json.dump(manifest, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n{len(manifest)} 单元 / {sum(m["pages"] for m in manifest)} 条')
    print('->', out)


main()
