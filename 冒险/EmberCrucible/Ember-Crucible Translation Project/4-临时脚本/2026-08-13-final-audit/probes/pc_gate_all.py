# -*- coding: utf-8 -*-
"""人物一致性镜头 G：**全体具名 NPC 的英文闸**。

权威名表 = `*.actors.<X>.name`（中文剥双语尾巴）。对每个 NPC：
  · 英文闸：叶的英文里出现该 NPC 英文名（全名 / 姓 / 名，词边界）
  · 中文侧检查：是否出现该 NPC 的权威中文名（或其姓/名片段，>=2 字）
  英文出现而中文一个片段都没有 -> 报 MISS（可能改写成代称，也可能音译分裂）

输出按 NPC 汇总，附缺失叶的中文原文片段，供人判读。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all
from pc_align import plain

CN_TAIL_EN = re.compile(r'\s*[A-Za-z0-9 \'’,.()"“”-]+$')
STOPWORDS = {'the', 'of', 'and', 'a', 'an', 'party', 'guard', 'captain', 'sage',
             'chess', 'sin', 'cap', 'mug', 'kern', 'meri', 'leeph', 'tauric', 'clipper'}


def main():
    rows = [r for r in load_all() if r[4]]
    auth = {}
    for repo, pack, path, en, cn in rows:
        m = re.match(r'^[^.]+\.actors\.([^.]+)\.name$', path)
        if not m:
            continue
        z = CN_TAIL_EN.sub('', cn).strip()
        if re.search(r'[一-鿿]', z):
            auth.setdefault(m.group(1), z)
    # 只留「像人名」的：至少两个词、首字母大写、不含通用词
    people = {}
    for en_name, cn_name in auth.items():
        toks = en_name.split()
        if len(toks) < 2:
            continue
        if not all(t[:1].isupper() for t in toks if t[:1].isalpha()):
            continue
        if any(t.lower() in ('guard', 'protector', 'warrior', 'adventurer', 'scout',
                             'veteran', 'soldier', 'raider', 'boarder', 'crew',
                             'construct', 'dragon', 'drake', 'juvenile', 'swarm')
               for t in toks):
            continue
        people[en_name] = cn_name
    print('候选 NPC（actor 名，双词以上）', len(people))

    report = []
    for en_name, cn_name in sorted(people.items()):
        # 只闸**全名**（整串），不闸单个词 —— 单词会把 Arcturian / Ordani 这类文化词全兜进来
        rx = re.compile(r'\b' + re.escape(en_name).replace(r'\ ', r'\s+') + r'\b')
        cn_frags = [f for f in re.split(r'[·"“”\s]+', cn_name) if len(f) >= 2]
        cn_frags = cn_frags or [cn_name]
        miss = []
        nhit = 0
        for repo, pack, path, en, cn in rows:
            pe = plain(en)
            if not rx.search(pe):
                continue
            pc = plain(cn)
            if any(f in pc for f in cn_frags):
                nhit += 1
                continue
            miss.append((repo, pack, path, pe, pc))
        if miss:
            report.append({'en': en_name, 'cn': cn_name, 'hit': nhit,
                           'miss': len(miss), 'rows': miss})
    report.sort(key=lambda r: -r['miss'])
    print('有缺失叶的 NPC', len(report))
    for r in report[:60]:
        print(f"\n### {r['en']} = {r['cn']}   命中 {r['hit']} 叶 / 缺失 {r['miss']} 叶")
        for repo, pack, path, pe, pc in r['rows'][:4]:
            print(f"   - {pack[:16]} {path[-62:]}")
            m = re.search(r'\b' + re.escape(r['en'].split()[-1]) + r'\b', pe)
            i = m.start() if m else 0
            print(f"     EN …{pe[max(0,i-90):i+90]}…")
            print(f"     CN …{pc[:170]}…")
    json.dump([{k: v for k, v in r.items() if k != 'rows'} |
               {'rows': [(a, b, c) for a, b, c, _, _ in r['rows']]} for r in report],
              open('pc_gate_all.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
