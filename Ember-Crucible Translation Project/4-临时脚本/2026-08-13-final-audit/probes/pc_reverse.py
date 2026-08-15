# -*- coding: utf-8 -*-
"""人物一致性镜头 I：**反向闸** —— 中文点了英文没点的人名（潜在泄底/张冠李戴）。

对每个权威 NPC（actor name 双语对），若某叶中文出现该 NPC 的中文名，
而该叶英文里既没有其英文全名、也没有指向该 actor 的 @UUID，则报。
典型成因：
  1. 英文用代称（"the necromancer"），中文把真名点出来 -> **泄底**（第十轮 Vinarith 类）
  2. 中文把 A 写成了 B（张冠李戴）
  3. 英文用 [[lookup @name]] 宏，中文照抄（不算，已排除：中文里也有该宏时跳过）
"""
from __future__ import annotations
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all
from pc_align import plain

CN_TAIL_EN = re.compile(r'\s*[A-Za-z0-9 \'’,.()"“”-]+$')
UUID_ANY = re.compile(r'@(?:UUID|Embed)\[[^\]]*\]')


def main():
    rows = [r for r in load_all() if r[4]]
    auth = {}
    for repo, pack, path, en, cn in rows:
        m = re.match(r'^[^.]+\.actors\.([^.]+)\.name$', path)
        if not m:
            continue
        z = CN_TAIL_EN.sub('', cn).strip()
        toks = m.group(1).split()
        if len(toks) >= 2 and len(z) >= 4 and re.search(r'[一-鿿]', z) and '·' in z:
            auth[z] = m.group(1)
    print('权威人名（带·的双词 actor）', len(auth))
    out = []
    for repo, pack, path, en, cn in rows:
        if path.endswith('.name') or path.endswith('.tokenName'):
            continue
        pc = plain(cn)
        if not pc:
            continue
        pe = plain(en)
        for z, e in auth.items():
            if z not in pc:
                continue
            surname = e.split()[-1]
            given = e.split()[0]
            if re.search(r'\b(' + re.escape(surname) + '|' + re.escape(given) + r')\b', pe):
                continue
            if UUID_ANY.search(en) or '[[lookup' in en:
                continue
            i = pc.find(z)
            out.append({'repo': repo, 'pack': pack, 'path': path, 'cn_name': z,
                        'en_name': e, 'cn_ctx': pc[max(0, i - 110):i + 110],
                        'en_head': pe[:260]})
    print('中文点名而英文未点名的叶', len(out))
    for r in out[:50]:
        print(f"\n### {r['cn_name']}({r['en_name']}) @ {r['pack'][:16]} {r['path'][-64:]}")
        print('   CN …', r['cn_ctx'])
        print('   EN …', r['en_head'])
    json.dump(out, open('pc_reverse.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
