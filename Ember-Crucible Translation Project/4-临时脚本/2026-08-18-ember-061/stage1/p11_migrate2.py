# -*- coding: utf-8 -*-
"""P11: 迁移识别（收严版）。
S1 改名/移位：路径同长、恰一个段不同（下标 i），且 **old 段 gs[i] 在 061 的同一父下已经不存在**
     （否则就只是「同一个通用法术恰好也长在别的 actor 身上」——那不是迁移）。值相似 >= 0.90。
S2 结构下沉：a == g + ['public'] 之类，值相似 >= 0.90。
S3 结构上提。
S4 同父键集 1↔1 改名（levels / effects 之类）：父下 gone 键与 added 键各恰好配对，用文本相似度择优，
   不设 0.90 门槛（改名本来就会改字面），但要求 old 段真的没了、new 段真的是新的。
"""
import os, sys, difflib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
out = {}
for f in FILES:
    gone, added = delta[f]["gone"], delta[f]["added"]
    if not gone:
        out[f] = {"migrate": [], "delete": []}
        continue
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))
    A = [(p, split_path(p), get_at(en061, split_path(p))) for p in added]
    by_len = collections.defaultdict(list)
    for p, sp, v in A:
        by_len[len(sp)].append((p, sp, v))

    cands = []
    for g in gone:
        gs = split_path(g)
        vg = get_at(en060, gs)
        if not isinstance(vg, str):
            continue
        for p, sp, v in by_len.get(len(gs), []):
            diff = [i for i in range(len(gs)) if gs[i] != sp[i]]
            if len(diff) != 1 or not isinstance(v, str):
                continue
            i = diff[0]
            # old 段必须在 061 里真的没了
            if has_at(en061, gs[:i + 1]):
                continue
            r = 1.0 if v == vg else difflib.SequenceMatcher(None, vg, v).ratio()
            if r >= 0.90:
                cands.append((r, 'S1', g, p, str(gs[i]) + '->' + str(sp[i])))
        for p, sp, v in by_len.get(len(gs) + 1, []):
            if sp[:len(gs)] != gs or not isinstance(v, str):
                continue
            r = 1.0 if v == vg else difflib.SequenceMatcher(None, vg, v).ratio()
            if r >= 0.90:
                cands.append((r, 'S2', g, p, '+' + str(sp[-1])))
        for p, sp, v in by_len.get(len(gs) - 1, []):
            if gs[:len(sp)] != sp or not isinstance(v, str):
                continue
            r = 1.0 if v == vg else difflib.SequenceMatcher(None, vg, v).ratio()
            if r >= 0.90:
                cands.append((r, 'S3', g, p, '-' + str(gs[-1])))

    # S4: 同父 1<->1 键改名
    gpar = collections.defaultdict(list)
    apar = collections.defaultdict(list)
    for g in gone:
        gs = split_path(g)
        for i in range(2, len(gs)):
            gpar[('/'.join(str(x) for x in gs[:i]), str(gs[i]))].append(g)
    for p, sp, v in A:
        for i in range(2, len(sp)):
            apar[('/'.join(str(x) for x in sp[:i]), str(sp[i]))].append(p)
    parents = collections.defaultdict(lambda: [set(), set()])
    for (par, key) in gpar:
        parents[par][0].add(key)
    for (par, key) in apar:
        parents[par][1].add(key)
    for par, (gk, ak) in parents.items():
        # 只看父容器在两版都存在、且 gone 键在 061 全没、added 键在 060 全没
        gk = {k for k in gk if not has_at(en061, split_path('/' + par + '/' + k))}
        ak = {k for k in ak if not has_at(en060, split_path('/' + par + '/' + k))}
        if len(gk) == 1 and len(ak) == 1:
            gkey, akey = list(gk)[0], list(ak)[0]
            for g in gpar[(par, gkey)]:
                gs = split_path(g)
                i = len(split_path('/' + par))
                sp = gs[:i] + [akey] + gs[i + 1:]
                p = '/' + '/'.join(str(x) for x in sp)
                if p in added:
                    vg = get_at(en060, gs); v = get_at(en061, sp)
                    if isinstance(vg, str) and isinstance(v, str):
                        r = difflib.SequenceMatcher(None, vg, v).ratio()
                        cands.append((r, 'S4', g, p, f"{gkey}->{akey}"))

    order = {'S1': 0, 'S2': 0, 'S3': 0, 'S4': 1}
    cands.sort(key=lambda x: (order[x[1]], -x[0]))
    ug, ua, mig = set(), set(), []
    for r, s, g, p, extra in cands:
        if g in ug or p in ua:
            continue
        ug.add(g); ua.add(p)
        mig.append({"src": g, "dst": p, "r": round(r, 4), "shape": s, "at": extra})
    out[f] = {"migrate": mig, "delete": [g for g in gone if g not in ug]}
    print(f"\n##### {f}: 迁移 {len(mig)} / 真删 {len(out[f]['delete'])}")
    for m in sorted(mig, key=lambda x: x['src']):
        print(f"  [{m['shape']} r={m['r']} {m['at']}]\n      {m['src'].split('/',3)[-1]}\n   -> {m['dst'].split('/',3)[-1]}")

t = sum(len(out[f]['migrate']) for f in FILES)
dd = sum(len(out[f]['delete']) for f in FILES)
print(f"\n合计 迁移 {t} / 真删 {dd} / 总 {t+dd}")
assert t + dd == 314
jdump(out, os.path.join(WORK, 'migrate.json'))
