#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P8：Crucible/Ember **资源类数值**在不同页是否一致（Action / Focus / Boon / …）。

镜头要求「同一数值 + 同一机制词在不同页重复陈述时中文是否自相矛盾」。
本探针对每个 (机制词, 数值) 组合做全库普查：
* **块内对照**：EN 块里出现 `6 Action`，CN 同块里 `动作` 附近的数是不是 6；
* **跨页普查**：同一机制词在全库出现过哪些数值、各在哪些页，人工一眼看出谁跟谁打架。

机制词表来自 crucible 系统的资源/骰池名词，只收**中文译名唯一**的（避免同形异义噪声）。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import CJK, plain, split_blocks, load_all, EN_WORD_NUM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)

# (英文词, 中文词)。中文只写**唯一**译名，多义的不收。
TERMS = [
    ('Action', '动作'), ('Focus', '专注'), ('Boons?', '恩惠骰'), ('Banes?', '祸骰'),
    ('Heroism', '英雄气概'), ('Morale', '士气'), ('Health', '生命值'),
    ('Talent Points?', '天赋点'), ('Milestone Points?', '里程碑点数'),
    ('Milestones?', '里程碑'), ('Threat', '威胁'), ('Stride', '步幅'),
    ('Engagement', '交战'), ('Wounds?', '创伤'), ('Madness', '疯狂'),
    ('Resolve', '决意'), ('Initiative', '先攻'), ('Advantage', '优势'),
    ('Disadvantage', '劣势'), ('Exhaustion', '力竭'),
]
WORDS = '|'.join(sorted(EN_WORD_NUM, key=len, reverse=True))
NUMTOK = r'(?:\d+|' + WORDS + r')'


def val(tok):
    t = tok.lower()
    return t if t.isdigit() else str(EN_WORD_NUM.get(re.sub(r's$', '', t), ''))


def page_of(path):
    p = path.split('.')
    return f'{p[1]}/{p[3]}' if len(p) >= 4 and p[2] == 'pages' else '.'.join(p[:2])


def main():
    census = defaultdict(Counter)      # 英文词 -> Counter(值)
    where = defaultdict(lambda: defaultdict(set))
    mism = []
    pats = [(re.compile(r'(?<![\w.])(' + NUMTOK + r')\s+(?:points? of\s+)?(' + en + r')\b'),
             re.compile(r'\b(' + en + r')\s*(?:pool)?\s*(?:of|:|=)?\s*(' + NUMTOK + r')(?![\w])'),
             en, cn) for en, cn in TERMS]
    for repo, pack, path, en_s, cn_s in load_all():
        if not cn_s or not CJK.search(cn_s):
            continue
        eb, cb = split_blocks(en_s), split_blocks(cn_s)
        if len(eb) != len(cb):
            continue
        for i, (e, c) in enumerate(zip(eb, cb)):
            pe, pc = plain(e), plain(c)
            if not pe or not CJK.search(pc):
                continue
            for pa, pb, en_t, cn_t in pats:
                found = [(val(m.group(1)), m.group(2)) for m in pa.finditer(pe)]
                found += [(val(m.group(2)), m.group(1)) for m in pb.finditer(pe)]
                for v, _ in found:
                    if not v or v == '1':
                        continue
                    census[en_t][v] += 1
                    where[en_t][v].add(f'{repo}:{page_of(path)}')
                    # CN 同块里该词附近的数
                    cnvals = set()
                    for m in re.finditer(r'(?<![\d.])(\d+)\s*(?:点|个)?\s*' + cn_t, pc):
                        cnvals.add(m.group(1))
                    for m in re.finditer(cn_t + r'\s*(?:池)?\s*(?:为|是|：|:)?\s*(\d+)', pc):
                        cnvals.add(m.group(1))
                    if cnvals and v not in cnvals:
                        mism.append({'repo': repo, 'pack': pack, 'path': path,
                                     'page': page_of(path), 'block': i,
                                     'term': en_t, 'en_val': v, 'cn_vals': sorted(cnvals),
                                     'en': pe[:350], 'cn': pc[:350]})
    seen, uniq = set(), []
    for r in mism:
        k = (r['page'], r['block'], r['term'], r['en_val'], r['en'][:120])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f'块内 (机制词, 数值) 中英不符：{len(mism)}（孪生去重后 {len(uniq)}）')
    for r in uniq[:40]:
        print('\n' + '-' * 88)
        print(f'{r["term"]}: EN={r["en_val"]} CN={r["cn_vals"]}  @ {r["repo"]}/{r["page"]} [b{r["block"]}]')
        print('   EN:', r['en'][:280])
        print('   CN:', r['cn'][:280])
    print('\n===== 跨页普查：每个机制词出现过的数值 =====')
    for t in census:
        items = census[t].most_common()
        print(f'\n{t}: ' + ', '.join(f'{v}×{n}' for v, n in items))
        for v, n in items:
            if n <= 6:
                print(f'    {v}: {sorted(where[t][v])[:6]}')
    json.dump({'mismatch': uniq,
               'census': {t: {v: sorted(where[t][v]) for v in census[t]} for t in census}},
              open(os.path.join(OUT, 'xp_p8_resource_num.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
