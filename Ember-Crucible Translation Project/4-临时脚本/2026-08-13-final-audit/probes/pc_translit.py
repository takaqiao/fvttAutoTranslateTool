# -*- coding: utf-8 -*-
"""人物一致性镜头 F：**同一个 NPC 的中文音译在不同页分裂**。

现有判据的盲区
--------------
* `scan_name_splits` 只比 `name` 字段；
* `scan_same_en_split` 要求**整条英文串逐字相同**才聚合 —— 同一个人名出现在两段
  完全不同的散文里时，它看不见；
* `scan_renamed_terms` / `unify_terms` 按术语表跑，术语表里根本没有这些次要 NPC。
实证：`Nathira Jessos` 在地名志作「纳西拉·杰索斯」、在任务页作「纳希拉·杰索斯」，
三类判据全部沉默。

做法
----
1. 全库中文侧抽出「音译人名」形态 token：`汉字{1,6}·汉字{1,8}`（本库人名一律用「·」）。
2. 两两比较：**同长度且只差 1 个汉字**（或去掉「·」后编辑距离 1）的算候选分裂对。
3. **英文闸**：对每个候选对，取含变体 A 的叶子的英文侧、含变体 B 的叶子的英文侧，
   求两边英文里都出现的「大写词序列」交集。交集非空 = 同一个英文名两种中文，报。
   交集为空 = 两个不同的人（如「德拉肯」vs「德拉贡」），不报。

假阳性来源：两个英文名本身就相近（如 Nir'ae / Nirae）；同一叶里同时提到两人。
所以输出附英文交集与出处，供人复核。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all
from pc_align import plain

NAME = re.compile(r'[一-鿿]{1,6}·[一-鿿]{1,8}(?:·[一-鿿]{1,8})?')
CAPSEQ = re.compile(r"\b[A-Z][A-Za-z'’À-ɏ-]+(?:\s+[A-Z][A-Za-z'’À-ɏ-]+){0,3}")


def edit1(a, b):
    """长度相等且恰好差 1 个字符，或长度差 1 且是插入/删除。"""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(t)):
        if t[:i] + t[i + 1:] == s:
            return True
    return False


def main():
    rows = [r for r in load_all() if r[4]]
    occ = defaultdict(list)     # cn name -> [row index]
    for i, (repo, pack, path, en, cn) in enumerate(rows):
        for nm in set(NAME.findall(plain(cn))):
            occ[nm].append(i)
    names = sorted(occ)
    print('中文音译人名 token 数', len(names))

    # 候选对
    cands = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if edit1(a, b):
                cands.append((a, b))
    print('形近候选对', len(cands))

    out = []
    for a, b in cands:
        ea = Counter()
        for i in occ[a]:
            ea.update(set(CAPSEQ.findall(plain(rows[i][3]))))
        eb = Counter()
        for i in occ[b]:
            eb.update(set(CAPSEQ.findall(plain(rows[i][3]))))
        common = sorted(set(ea) & set(eb), key=lambda w: -len(w))
        # 只留「像人名」的交集词：>=2 词或 >=6 字母
        common = [w for w in common if ' ' in w or len(w) >= 6]
        if not common:
            continue
        out.append({'a': a, 'b': b, 'na': len(occ[a]), 'nb': len(occ[b]),
                    'common': common[:12],
                    'pa': [rows[i][2][-58:] for i in occ[a]][:6],
                    'pb': [rows[i][2][-58:] for i in occ[b]][:6]})
    out.sort(key=lambda r: -(min(r['na'], r['nb'])))
    print('英文闸后候选', len(out))
    for r in out:
        print(f"\n### {r['a']}({r['na']}叶) / {r['b']}({r['nb']}叶)")
        print('   共同英文:', r['common'])
        print('   A:', r['pa'])
        print('   B:', r['pb'])
    json.dump(out, open('pc_translit.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
