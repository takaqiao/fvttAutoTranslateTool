#!/usr/bin/env python3
"""按**英文闸**在对齐字幕语料里给一个术语投票，并按**血脉**而不是按文件计票。

  python term_vote.py "<english regex>" [--zh <regex>] [--show N]

为什么不是裸词频
----------------
PROJECT.md §3.3：绝不从裸中文词频得结论。这里先用英文正则筛出命中行，
再统计那些行的中文侧写法。没有英文闸的计数既会漏也会误伤。

为什么按血脉计票
----------------
CnSCG 那四部（异形 1/2/3/4）是同一条翻译血脉，跨片"一致"是共同出处的产物，
必须折成**一票**。Romulus / Covenant / Earth / Prometheus 各自独立。
输出同时给「行数」和「血脉数」——**以血脉数为准**。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.abspath(os.path.join(HERE, '..', '..', '7-其他内容', 'reference', 'subtitles'))

# 同一条翻译血脉折成一票
LINEAGE = [
    (re.compile(r'CnSCG|RARBG', re.I), 'CnSCG(异形1/2/3/4·同源)'),
    (re.compile(r'Romulus', re.I), 'Romulus2024'),
    (re.compile(r'Covenant', re.I), 'Covenant2017'),
    (re.compile(r'Earth', re.I), 'AlienEarth2025'),
    (re.compile(r'Prometheus', re.I), 'Prometheus2012'),
]


def lineage_of(name):
    for pat, label in LINEAGE:
        if pat.search(name):
            return label
    return 'other:' + name


def load():
    """去重：同一血脉里内容相同的文件（如 chs&eng / eng&chs）只留一份。"""
    seen, rows = set(), []
    manifest_path = os.path.join(CORPUS, '_manifest.json')
    if not os.path.exists(manifest_path):
        sys.exit(f'corpus not built: {CORPUS}\nrun build_subtitle_corpus.py first')
    manifest = json.load(open(manifest_path, encoding='utf-8'))
    for rel, meta in manifest.items():
        path = os.path.join(CORPUS, meta['tsv'])
        if not os.path.exists(path):
            continue
        lin = lineage_of(rel)
        pairs = []
        for line in open(path, encoding='utf-8'):
            if '\t' not in line:
                continue
            en, zh = line.rstrip('\n').split('\t', 1)
            pairs.append((en, zh))
        sig = (lin, len(pairs), pairs[0] if pairs else None)
        if sig in seen:
            continue
        seen.add(sig)
        rows += [(lin, rel, en, zh) for en, zh in pairs]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('english', help='英文侧正则（英文闸）')
    ap.add_argument('--zh', help='可选：只统计中文侧命中此正则的行')
    ap.add_argument('--show', type=int, default=8, help='打印几条样例')
    args = ap.parse_args()

    en_re = re.compile(args.english, re.I)
    zh_re = re.compile(args.zh) if args.zh else None

    rows = load()
    hits = [r for r in rows if en_re.search(r[2]) and (not zh_re or zh_re.search(r[3]))]
    print(f'语料 {len(rows)} 对 / 英文闸命中 {len(hits)} 行\n')
    if not hits:
        print('  (零命中 — 这条术语在影视语料里没有依据，不要据此下结论)')
        return

    # 中文侧候选：抽出连续汉字块，按血脉计票
    by_cand = defaultdict(set)
    counts = Counter()
    for lin, _rel, _en, zh in hits:
        for chunk in re.findall(r'[一-鿿]{2,}', zh):
            by_cand[chunk].add(lin)
            counts[chunk] += 1

    print(f'{"候选":<16}{"血脉":>4}{"行数":>6}  出处')
    print('-' * 72)
    for cand, lins in sorted(by_cand.items(), key=lambda kv: (-len(kv[1]), -counts[kv[0]]))[:18]:
        print(f'{cand:<16}{len(lins):>4}{counts[cand]:>6}  {"·".join(sorted(lins))}')

    print(f'\n--- 样例（{min(args.show, len(hits))} / {len(hits)}）---')
    for lin, _rel, en, zh in hits[:args.show]:
        print(f'  [{lin}]\n    EN {en[:110]}\n    ZH {zh[:110]}')


if __name__ == '__main__':
    main()
