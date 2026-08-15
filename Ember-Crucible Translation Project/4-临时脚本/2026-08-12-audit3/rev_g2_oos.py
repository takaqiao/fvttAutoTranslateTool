# -*- coding: utf-8 -*-
"""G2 复核：把闸判 OUT_OF_SCOPE 的 114 条表结果（目标在 dnd5e.items / dnd5e.equipment24）
拿本机已安装的 `dnd-simplified-chinese-babele-patch` 对一遍。

G2 自称这批「本项目无从核对」。但该模块就装在本机，它给 dnd5e 物品的中文名是玩家
实际会看到的窗口标题 —— 我们表结果行的中文若与它不同，就是这条闸要抓的那个症状。
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "3-常用脚本", "qa"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import scan_name_binding as G

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
S = (r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt"
     r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad")
D5 = (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules"
      r"\dnd-simplified-chinese-babele-patch\translation\cn")

# dnd5e 侧译文：{英文名 -> Counter(中文名)}
d5 = collections.defaultdict(collections.Counter)
for fn in ("dnd5e.items.json", "dnd5e.equipment24.json"):
    doc = json.load(open(os.path.join(D5, fn), encoding="utf-8"))
    for k, v in (doc.get("entries") or {}).items():
        if isinstance(v, dict):
            nm = v.get("name")
            if isinstance(nm, str) and nm:
                d5[k][nm] += 1
        elif isinstance(v, str):
            d5[k][v] += 1
print(f"dnd5e patch 条目 {len(d5)}")

repos = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
packs = G.load_packs(repos)
ids, notes, results = G.load_bindings([os.path.join(S, "audit3", f) for f in
                                       ("re_bind_ember.json", "re_bind_crucible.json",
                                        "re_bind_dnd5e.json")])
res_bind = collections.defaultdict(list)
for r in results:
    res_bind[(f"{r.get('pkg')}.{r.get('pack')}", r.get("adventure"),
              r.get("table"), r.get("range"))].append(r)

rows = []
for packkey, pv in sorted(packs.items()):
    en, cn = pv["en"], pv["cn"]
    for ent, ev in (en.get("entries") or {}).items():
        cv = (cn.get("entries") or {}).get(ent) or {}
        for tname, tv in (ev.get("tables") or {}).items():
            ctv = ((cv.get("tables") or {}).get(tname) or {})
            for rng, rv in (tv.get("results") or {}).items():
                if not isinstance(rv, dict):
                    continue
                en_label = rv.get("name")
                if not isinstance(en_label, str) or not en_label.strip():
                    continue
                cn_label = ((ctv.get("results") or {}).get(rng) or {}).get("name")
                binds = res_bind.get((packkey, ent, tname, rng)) or \
                        res_bind.get((packkey, None, tname, rng)) or []
                for b in binds:
                    u = b.get("documentUuid") or ""
                    if not u.startswith("Compendium.dnd5e."):
                        continue
                    tpack, tid = G.parse_uuid(u, packkey)
                    t = G.resolve(ids, tid, *(tpack.split(".", 1) + [None])[:2])
                    tgt_en = (t or {}).get("name")
                    if not tgt_en or en_label.strip() != tgt_en.strip():
                        continue          # BY_DESIGN 别名，不在此列
                    d5cn = dict(d5.get(tgt_en) or {})
                    rows.append({"pack": pv["file"],
                                 "path": f"{ent}.tables.{tname}.results.{rng}.name",
                                 "en": en_label, "our_cn": cn_label,
                                 "target_pack": tpack, "d5_cn": d5cn,
                                 "match": bool(d5cn) and cn_label in d5cn})

miss = [r for r in rows if r["d5_cn"] and not r["match"]]
nod5 = [r for r in rows if not r["d5_cn"]]
print(f"OUT_OF_SCOPE 行 {len(rows)}；dnd5e 侧有中文名的 {len(rows)-len(nod5)}；"
      f"其中我方中文 != dnd5e 中文 = {len(miss)}；dnd5e 侧也没中文 {len(nod5)}")
for r in miss:
    print(f"  {r['pack']} :: {r['path']}")
    print(f"     EN {r['en']!r}  我方 {r['our_cn']!r}  dnd5e {list(r['d5_cn'])}")
print("\n--- dnd5e 侧也没中文名的 ---")
for r in nod5:
    print(f"  {r['pack']} :: {r['en']!r} -> 我方 {r['our_cn']!r} ({r['target_pack']})")

json.dump({"rows": rows, "mismatch": miss, "no_d5_name": nod5},
          open(os.path.join(S, "rev_g2", "oos.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
