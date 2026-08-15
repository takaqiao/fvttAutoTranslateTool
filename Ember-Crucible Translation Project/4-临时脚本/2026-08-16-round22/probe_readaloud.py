#!/usr/bin/env python3
"""现读两仓，把每一段增强器可见文本（readaloud / label）的实测量摊开。

反空转：先打印「我这次读了多少叶 / 配了多少对 / 摊出多少槽」，说不出来就不算数。
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, '3-常用脚本', 'qa'))

import assert_resolutions as AR   # noqa: E402

CJK = re.compile(r'[一-鿿]')


def load_gloss():
    gp = os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.json')
    g = {}
    for k, v in json.load(open(gp, encoding='utf-8')).items():
        zh = v.split(' ')[0].strip() if isinstance(v, str) else ''
        if len(k) >= 5 and len(zh) >= 2 and CJK.search(zh) and not re.search(r'[A-Za-z]', zh):
            g[k] = zh
    return g


def main():
    repos = {}
    for name, rel in AR.REPOS.items():
        d = os.path.join(P, rel)
        if os.path.isdir(d):
            repos[name] = d
    ctx = AR.Ctx(repos, {})
    gloss = load_gloss()
    print(f'词表锚点 {len(gloss)} 条')

    n_leaf = n_pair = n_unpaired = 0
    rows = []
    for repo, pack, path, ev, cv in ctx.all_pairs(None):
        if '@' not in ev and '@' not in cv:
            continue
        n_leaf += 1
        pairs, unp = AR._enr_pairs(ev, cv, AR._ENR_TEXT_PARAMS)
        n_pair += len(pairs)
        n_unpaired += unp
        for tgt, es, cs in pairs:
            for slot in set(es) | set(cs):
                et, ct = es.get(slot), cs.get(slot)
                rows.append(dict(repo=repo, pack=pack, path=path, tgt=tgt, slot=slot,
                                 en=et, cn=ct))
    print(f'读了 {sum(len(v) for v in ctx.pairs.values())} 对叶 · 含 @ 的 {n_leaf} 叶 · '
          f'配对增强器 {n_pair} 个（配不上 {n_unpaired}）· 摊出槽 {len(rows)} 个')

    from collections import Counter
    print('槽类型分布:', Counter(r['slot'] for r in rows))

    ra = [r for r in rows if r['slot'] == 'param:readaloud']
    print(f'\nreadaloud 槽 {len(ra)} 个；两侧都有的 {sum(1 for r in ra if r["en"] and r["cn"])}')
    both = [r for r in ra if r['en'] and r['cn']]
    for r in both:
        e, c = r['en'], r['cn']
        r['en_len'], r['cn_len'] = len(e), len(c)
        r['ratio'] = round(len(c) / max(len(e), 1), 3)
        r['cn_cjk'] = len(CJK.findall(c))
        hits = [(k, v) for k, v in gloss.items()
                if re.search(r'\b' + re.escape(k) + r'\b', e)]
        r['anchor_hits'] = len(hits)
        r['anchor_miss'] = [f'{k}->{v}' for k, v in hits if v not in c]
    both.sort(key=lambda r: r['ratio'])
    print(f'\nEN 总字符 {sum(r["en_len"] for r in both)} / CN 总字符 {sum(r["cn_len"] for r in both)}')
    print(f'比值 min {both[0]["ratio"]} / max {both[-1]["ratio"]} / '
          f'中位 {both[len(both)//2]["ratio"]}')
    print(f'命中词表锚点的段 {sum(1 for r in both if r["anchor_hits"])} 段 · '
          f'缺定译的段 {sum(1 for r in both if r["anchor_miss"])} 段 · '
          f'缺定译总条数 {sum(len(r["anchor_miss"]) for r in both)}')
    print('\n比值最低的 12 段:')
    for r in both[:12]:
        print(f'  {r["ratio"]:.3f}  EN{r["en_len"]:4d}/CN{r["cn_len"]:3d} 汉{r["cn_cjk"]:3d} '
              f'锚{r["anchor_hits"]:2d} 缺{len(r["anchor_miss"])}  '
              f'{r["repo"]}/{r["path"][-60:]}')
    print('\n有缺定译的段:')
    for r in both:
        if r['anchor_miss']:
            print(f'  {r["repo"]}/{r["path"][-70:]}  {r["anchor_miss"][:8]}')

    # label 槽同样摊一遍（作为对照，不一定纳入本闸）
    lb = [r for r in rows if r['slot'] == 'param:label' and r['en'] and r['cn']]
    print(f'\nlabel 槽两侧都有的 {len(lb)} 个；EN 字符 {sum(len(r["en"]) for r in lb)} / '
          f'CN 字符 {sum(len(r["cn"]) for r in lb)}')

    out = os.path.join(HERE, 'readaloud_probe.json')
    json.dump({'slots': len(rows), 'readaloud_both': len(both),
               'rows': [{k: v for k, v in r.items()} for r in both]},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
