# -*- coding: utf-8 -*-
"""人物一致性镜头 D：**代词标注字段**（she/her、he/him、they/them）的处理是否统一。

英文源在人物条里用 `Name (Alignment, Ancestry, pronouns)` 标注代词，全库量很大。
本库既定处理是**逐字保留英文代词对**（大多数如此）。本探针逐叶比对
EN 里各代词 token 的出现次数与 CN 里同 token 的出现次数，找出被改写/丢失的叶子，
并把 CN 侧实际写法抓出来分类。

假阳性：英文正文散文里出现的 "he/him" 之类（极少）；CN 把同一句改写但语义等价。
所以输出会带上下文原文供人判读。
"""
from __future__ import annotations
import json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_align import plain

TOKS = ['she/her', 'he/him', 'they/them', 'it/its', 'he/they', 'she/they',
        'they/she', 'they/he', 'xe/xem', 'ze/zir', 'any pronouns', 'no pronouns']
RX = {t: re.compile(re.escape(t), re.I) for t in TOKS}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all


def main():
    rows = []
    tot = Counter()
    for repo, pack, path, en, cn in load_all():
        if not cn:
            continue
        pe, pc = plain(en), plain(cn)
        miss = {}
        for t, rx in RX.items():
            ne, nc = len(rx.findall(pe)), len(rx.findall(pc))
            tot[t] += ne
            if ne > nc:
                miss[t] = (ne, nc)
        if miss:
            ctxs = []
            for t in miss:
                for m in RX[t].finditer(pe):
                    ctxs.append('EN…' + pe[max(0, m.start() - 110):m.end() + 3].strip())
            rows.append({'repo': repo, 'pack': pack, 'path': path, 'miss': miss,
                         'ctx_en': ctxs[:8], 'cn_sample': pc[:0]})
    print('EN 代词标注 token 总数：', dict(tot), ' 合计', sum(tot.values()))
    print('中文侧 token 数少于英文的叶子：', len(rows))
    for r in rows:
        print(f"\n=== {r['repo']}/{r['pack']}/{r['path'][-80:]}  {r['miss']}")
        for c in r['ctx_en']:
            print('   ', c)
    json.dump(rows, open('pc_prontag.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
