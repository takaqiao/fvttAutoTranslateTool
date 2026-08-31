# -*- coding: utf-8 -*-
"""人物一致性镜头 F2：NPC 中文音译分裂（有权威名锚点的版本）。

比 pc_translit.py 严：先建**权威人名表**，再在全库找它的形近变体。

权威表来源（都是「这个名字的正式写法」）
  1. `*.actors.<X>.name` 的中文（剥双语尾巴）
  2. 人物条 `Name (Alignment, Ancestry, pronouns)` 里紧邻括号的中文名
     （只取「·」两侧各不超过 6/8 字的最长中文串，再按权威表回收）
  3. `journals.Notable Figures.pages.*` 页名中文

变体搜索
  对每个「·」出现位置，取左最多 6 个汉字 + 右最多 8 个汉字，枚举
  (左后缀 k, 右前缀 m) 组合作为候选串；保留与某个权威名**编辑距离 1** 的候选。

英文闸
  变体 V 与权威名 A 若确为同一人，则含 V 的叶子的英文侧应当出现 A 对应的英文名。
  所以对每个 (A, V) 取「含 A 的叶」与「含 V 的叶」英文侧大写词序列的交集，
  交集里含 A 的英文名 -> 判为**真分裂**；否则只列为待查。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all
from pc_align import plain, PRONS

HAN = r'[一-鿿]'
CAPSEQ = re.compile(r"\b[A-Z][A-Za-z'’À-ɏ-]+(?:\s+[A-Z][A-Za-z'’À-ɏ-]+){0,3}")
PR = re.compile(PRONS, re.I)
CN_TAIL_EN = re.compile(r'\s*[A-Za-z0-9 \'’,.’()-]+$')


def edit1(a, b):
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


def main():
    rows = [r for r in load_all() if r[4]]
    # ---- 权威表 ----
    auth = {}           # cn name -> en name
    for repo, pack, path, en, cn in rows:
        m = re.match(r'^[^.]+\.actors\.([^.]+)\.name$', path)
        if m and '·' in cn:
            z = CN_TAIL_EN.sub('', cn).strip()
            if '·' in z:
                auth.setdefault(z, m.group(1))
    # 人物条名
    for repo, pack, path, en, cn in rows:
        pe, pc = plain(en), plain(cn)
        me, mc = list(PR.finditer(pe)), list(PR.finditer(pc))
        if not me or len(me) != len(mc):
            continue
        for a, b in zip(me, mc):
            we, wc = pe[max(0, a.start() - 130):a.start()], pc[max(0, b.start() - 90):b.start()]
            ie = max(we.rfind(c) for c in '([（')
            ic = max(wc.rfind(c) for c in '([（')
            if ie < 0 or ic < 0:
                continue
            mz = re.search(HAN + r'{1,6}·' + HAN + r'{1,10}(?:·' + HAN + r'{1,10})?\s*$', wc[:ic].rstrip())
            me2 = re.search(r"([A-Z][A-Za-z'’À-ɏ.-]*(?:\s+[A-Z][A-Za-z'’À-ɏ.-]*){0,3})\s*$", we[:ie].rstrip())
            if mz and me2:
                auth.setdefault(mz.group(0).strip(), me2.group(1))
    print('权威中文人名', len(auth))

    # ---- 全库候选串 ----
    cand_leaves = defaultdict(set)   # cand -> leaf idx
    for i, (repo, pack, path, en, cn) in enumerate(rows):
        pc = plain(cn)
        for m in re.finditer('·', pc):
            j = m.start()
            L = re.search(HAN + r'{1,6}$', pc[max(0, j - 6):j])
            R = re.match(HAN + r'{1,10}', pc[j + 1:j + 11])
            if not L or not R:
                continue
            ls, rs = L.group(0), R.group(0)
            for k in range(1, len(ls) + 1):
                for mm in range(1, len(rs) + 1):
                    cand_leaves[ls[-k:] + '·' + rs[:mm]].add(i)

    auth_names = list(auth)
    variants = []
    for a in auth_names:
        for v, idxs in cand_leaves.items():
            if v in auth or not edit1(a, v):
                continue
            # v 必须真的出现在语料里（cand_leaves 已保证），且不能是某权威名的子串扩展
            variants.append((a, v, idxs))

    out = []
    a_leaves = {a: {i for i, r in enumerate(rows) if a in plain(r[4])} for a in
                {x[0] for x in variants}}
    for a, v, idxs in variants:
        la = a_leaves[a]
        # 去掉 v 只是 a 的错切（v 出现的叶里 a 也出现，且 v 计数 <= a 计数）
        only_v = {i for i in idxs if a not in plain(rows[i][4])}
        if not only_v:
            continue
        ena = Counter()
        for i in la:
            ena.update(set(CAPSEQ.findall(plain(rows[i][3]))))
        env = Counter()
        for i in only_v:
            env.update(set(CAPSEQ.findall(plain(rows[i][3]))))
        common = sorted(set(ena) & set(env), key=lambda w: -len(w))
        en_name = auth[a]
        hit = [w for w in common if en_name and (en_name in w or w in en_name)]
        out.append({'auth': a, 'en': en_name, 'var': v,
                    'n_auth_leaf': len(la), 'n_var_only_leaf': len(only_v),
                    'en_gate_hit': hit[:5], 'common': common[:8],
                    'var_paths': [rows[i][2][-62:] for i in sorted(only_v)][:8],
                    'auth_paths': [rows[i][2][-62:] for i in sorted(la)][:5]})
    out.sort(key=lambda r: (-len(r['en_gate_hit']), -r['n_var_only_leaf']))
    print('候选变体', len(out), ' 其中英文闸命中', sum(1 for r in out if r['en_gate_hit']))
    for r in out:
        flag = 'HIT ' if r['en_gate_hit'] else 'weak'
        print(f"\n[{flag}] 权威 {r['auth']}({r['en']}) {r['n_auth_leaf']}叶  vs 变体 {r['var']} {r['n_var_only_leaf']}叶")
        print('   英文闸:', r['en_gate_hit'], ' | 共同英文:', r['common'][:5])
        print('   变体出处:', r['var_paths'])
    json.dump(out, open('pc_translit2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
