# -*- coding: utf-8 -*-
"""G2 复核探针：独立重算闸的各档，专找假阴性。

三个方向：
A) OK 档里 `cand` 多值的（目标有 >1 个中文名）—— judge() 只要 `cn_label in cand`
   就算过，多值时可能匹配到「另一个同英文名文档」的中文，是结构性假阴性口子。
B) BY_DESIGN 档 —— 英文标签 != 目标英文名就直接放过。若英文只差冠词/大小写/标点，
   中文却完全不同，玩家仍会看到对不上。
C) NOT_BOUND / OUT_OF_SCOPE 档里的表结果名，做孪生包（ember.adventure vs
   ember.crucible-adventure）逐条比对：英文逐字相同、中文不同 = 观感级不一致。
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
     r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3")

repos = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
packs = G.load_packs(repos)
ids, notes, results = G.load_bindings([os.path.join(S, f) for f in
                                       ("re_bind_ember.json", "re_bind_crucible.json",
                                        "re_bind_dnd5e.json")])
indexes = {pk: G.build_name_index(v["en"], v["cn"]) for pk, v in packs.items()}

note_bind = collections.defaultdict(list)
for n in notes:
    note_bind[(f"{n.get('pkg')}.{n.get('pack')}", n.get("adventure"),
               n.get("scene"), n.get("text") or "")].append(n)
res_bind = collections.defaultdict(list)
for r in results:
    res_bind[(f"{r.get('pkg')}.{r.get('pack')}", r.get("adventure"),
              r.get("table"), r.get("range"))].append(r)

multi_ok, by_design, all_rows = [], [], []

for packkey, pv in sorted(packs.items()):
    en, cn = pv["en"], pv["cn"]
    idx = indexes[packkey]
    for ent_name, ev in (en.get("entries") or {}).items():
        cv = (cn.get("entries") or {}).get(ent_name) or {}

        # ---- 针脚 ----
        for sname, sv in (ev.get("scenes") or {}).items():
            csv_ = ((cv.get("scenes") or {}).get(sname) or {})
            for text, en_label in (sv.get("notes") or {}).items():
                if not isinstance(en_label, str):
                    continue
                cn_label = (csv_.get("notes") or {}).get(text)
                rows = note_bind.get((packkey, ent_name, sname, text)) or \
                       note_bind.get((packkey, None, sname, text)) or []
                bound = [r for r in rows if r.get("entryId") or r.get("pageId")]
                if rows and not bound:
                    continue
                targets, role = [], "pages"
                for row in bound:
                    t = None
                    if row.get("pageId"):
                        t = G.resolve(ids, row["pageId"], row.get("pkg"), row.get("pack"))
                    if t is None and row.get("entryId"):
                        t = G.resolve(ids, row["entryId"], row.get("pkg"), row.get("pack"))
                        if t is not None:
                            role = G.role_of(t.get("kind")) or "journals"
                    elif t is not None:
                        role = G.role_of(t.get("kind")) or "pages"
                    targets.append(t)
                have = [t for t in targets if t]
                target = have[0] if have else None
                if target is None:
                    continue
                tgt_pk = f"{target.get('pkg')}.{target.get('pack')}"
                cand = collections.Counter()
                cand.update((indexes.get(tgt_pk, {}).get(role) or {}).get(target.get("name") or "") or {})
                cand.update((idx.get(role) or {}).get(target.get("name") or "") or {})
                rec = {"kind": "note", "pack": pv["file"],
                       "path": f"{ent_name}.scenes.{sname}.notes.{text}",
                       "en": en_label, "cn": cn_label, "tgt_en": target.get("name"),
                       "tgt_pk": tgt_pk, "cand": dict(cand)}
                if en_label.strip() != (target.get("name") or "").strip():
                    by_design.append(rec)
                elif cn_label and cn_label in cand and len(cand) > 1:
                    multi_ok.append(rec)

        # ---- 表结果 ----
        for tname, tv in (ev.get("tables") or {}).items():
            ctv = ((cv.get("tables") or {}).get(tname) or {})
            for rng, rv in (tv.get("results") or {}).items():
                if not isinstance(rv, dict):
                    continue
                en_label = rv.get("name")
                cn_label = ((ctv.get("results") or {}).get(rng) or {}).get("name")
                if not isinstance(en_label, str) or not en_label.strip():
                    continue
                all_rows.append({"pack": pv["file"], "table": tname, "rng": rng,
                                 "en": en_label, "cn": cn_label,
                                 "path": f"{ent_name}.tables.{tname}.results.{rng}.name"})
                rows = res_bind.get((packkey, ent_name, tname, rng)) or \
                       res_bind.get((packkey, None, tname, rng)) or []
                bound = [r for r in rows if r.get("documentUuid") or r.get("documentId")]
                if rows and not bound:
                    continue
                targets, role = [], None
                for r in bound:
                    uuid = r.get("documentUuid")
                    if uuid:
                        tpack, tid = G.parse_uuid(uuid, packkey)
                    else:
                        tpack, tid = (r.get("documentCollection") or packkey), r.get("documentId")
                    t = G.resolve(ids, tid, *(tpack.split(".", 1) + [None])[:2]) if tid else None
                    if t:
                        role = G.role_of(t.get("kind"))
                        t = dict(t, packkey=tpack)
                    targets.append(t)
                have = [t for t in targets if t]
                target = have[0] if have else None
                if target is None:
                    continue
                tpk = target.get("packkey")
                cand = (indexes.get(tpk, idx).get(role or "entries") or {}).get(target.get("name") or "") or {}
                rec = {"kind": "result", "pack": pv["file"],
                       "path": f"{ent_name}.tables.{tname}.results.{rng}.name",
                       "en": en_label, "cn": cn_label, "tgt_en": target.get("name"),
                       "tgt_pk": tpk, "in_scope": tpk in packs, "cand": dict(cand)}

                if en_label.strip() != (target.get("name") or "").strip():
                    by_design.append(rec)
                elif tpk in packs and cn_label and cn_label in cand and len(cand) > 1:
                    multi_ok.append(rec)


def rec_path(e, t, r):
    return f"{e}.tables.{t}.results.{r}.name"


# ---- C) 孪生包表结果比对（含 NOT_BOUND / OUT_OF_SCOPE） ----
twin = collections.defaultdict(dict)
for r in all_rows:
    if r["pack"] in ("ember.adventure.json", "ember.crucible-adventure.json"):
        twin[(r["table"], r["en"])][r["pack"]] = r
conflicts = []
for key, side in twin.items():
    if len(side) == 2:
        a = side["ember.adventure.json"]["cn"]
        b = side["ember.crucible-adventure.json"]["cn"]
        if a != b:
            conflicts.append({"table": key[0], "en": key[1], "adv_cn": a, "cru_cn": b,
                              "adv_path": side["ember.adventure.json"]["path"],
                              "cru_path": side["ember.crucible-adventure.json"]["path"]})

out = {"multi_value_OK": multi_ok, "by_design": by_design, "twin_result_conflicts": conflicts,
       "counts": {"multi_value_OK": len(multi_ok), "by_design": len(by_design),
                  "twin_conflicts": len(conflicts), "all_result_rows": len(all_rows)}}
dst = os.path.join(os.path.dirname(S), "rev_g2", "probe.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out["counts"], ensure_ascii=False))
print("\n=== 孪生包表结果名冲突 ===")
for c in conflicts:
    print(f"  [{c['table']}] EN {c['en']!r}\n     adv {c['adv_cn']!r}\n     cru {c['cru_cn']!r}")
print("\n=== OK 但 cand 多值（假阴性口子） ===")
for m in multi_ok[:40]:
    print(f"  {m['pack']} :: {m['path']}\n     EN {m['en']!r} CN {m['cn']!r} tgt {m['tgt_en']!r}@{m['tgt_pk']} cand {m['cand']}")
print(f"\n-> {dst}")
