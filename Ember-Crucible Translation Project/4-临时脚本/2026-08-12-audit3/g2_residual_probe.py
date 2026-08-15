# -*- coding: utf-8 -*-
"""G2 复核探针：把 `scan_name_binding.py` **排除掉**的几个桶单独翻出来看。

闸本身只把 `BROKEN` 当缺陷（英文同名 + 目标有中文名 + 标签中文不同）。被它主动
排除的三个桶，各自都可能藏着别的类型的问题，本探针逐个查：

* `NOT_BOUND`（针脚没 entryId/pageId、结果行 type=text）—— 没有目标可点，但如果它的
  **英文**与某个文档英文名逐字相同，中文却写成另一个样子，仍是同一个专名的两种写法。
  不是功能缺陷，是术语一致性问题，另档报。
* `BY_DESIGN`（英文侧本来就不同名）—— 里面混着两种东西：真别名（`Corpse Loot
  (Arctus Plateau)`）和**英文只差一点点**（冠词/大小写/标点）的。后者中文若也差得很远，
  实际上是同一个专名的两种译法。用「英文包含关系 vs 中文包含关系」筛。
* `UNCERTAIN` —— 逐条打印，人工复核。

另外两项：
* 孪生包（`ember.adventure` / `ember.crucible-adventure`）里**英文逐字相同**的针脚/结果行，
  中文是否一致 —— 上一轮只修了 crucible 侧的尾巴，孪生侧可能没跟上。
* 闸的一个已知弱点：针脚目标的中文名只在**当前包**的索引里找，跨包目标会被误判成
  「目标没有中文名」。这里用全库索引复查，看有没有因此被藏起来的 BROKEN。
"""
import collections
import importlib.util
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
QA = os.path.join(P, "3-常用脚本", "qa", "scan_name_binding.py")
SCRATCH = (r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt"
           r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3")

spec = importlib.util.spec_from_file_location("snb", QA)
snb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snb)

REPOS = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
BINDS = [os.path.join(SCRATCH, f) for f in
         ("re_bind_ember.json", "re_bind_crucible.json", "re_bind_dnd5e.json")]

packs = snb.load_packs(REPOS)
ids, notes, results = snb.load_bindings(BINDS)
indexes = {pk: snb.build_name_index(v["en"], v["cn"]) for pk, v in packs.items()}

note_bind = collections.defaultdict(list)
for n in notes:
    note_bind[(f"{n.get('pkg')}.{n.get('pack')}", n.get("adventure"),
               n.get("scene"), n.get("text") or "")].append(n)
res_bind = collections.defaultdict(list)
for r in results:
    res_bind[(f"{r.get('pkg')}.{r.get('pack')}", r.get("adventure"),
              r.get("table"), r.get("range"))].append(r)


def norm(s):
    """英文标签归一化：小写、去冠词、去标点与空白。用来找『只差一点点』的别名。"""
    s = (s or "").lower()
    s = re.sub(r"\b(the|a|an|of)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def cn_core(s):
    """中文标签去掉双语并列的英文尾巴与括注，只留中文核心。"""
    s = (s or "")
    s = re.sub(r"[A-Za-z0-9'’\-\.\(\)/ ]+$", "", s).strip()
    return s


rows = []
for packkey, pv in sorted(packs.items()):
    en, cn = pv["en"], pv["cn"]
    idx = indexes[packkey]
    for ent, ev in (en.get("entries") or {}).items():
        cv = (cn.get("entries") or {}).get(ent) or {}

        # ---- 场景针脚 -------------------------------------------------
        for sname, sv in (ev.get("scenes") or {}).items():
            csv_ = (cv.get("scenes") or {}).get(sname) or {}
            for text, en_label in (sv.get("notes") or {}).items():
                if not isinstance(en_label, str):
                    continue
                cn_label = (csv_.get("notes") or {}).get(text)
                brows = note_bind.get((packkey, ent, sname, text)) or \
                        note_bind.get((packkey, None, sname, text)) or []
                bound = [b for b in brows if b.get("entryId") or b.get("pageId")]
                tgt, tpk = None, None
                for b in bound:
                    t = None
                    if b.get("pageId"):
                        t = snb.resolve(ids, b["pageId"], b.get("pkg"), b.get("pack"))
                    if t is None and b.get("entryId"):
                        t = snb.resolve(ids, b["entryId"], b.get("pkg"), b.get("pack"))
                    if t:
                        tgt = t
                        tpk = f"{t.get('pkg')}.{t.get('pack')}"
                        break
                rows.append({"kind": "note", "packkey": packkey, "pack": pv["file"],
                             "repo": os.path.basename(pv["repo"]),
                             "path": f"{ent}.scenes.{sname}.notes.{text}",
                             "en": en_label, "cn": cn_label,
                             "bound": bool(bound), "has_rows": bool(brows),
                             "tgt_en": (tgt or {}).get("name"), "tgt_pk": tpk,
                             "tgt_kind": (tgt or {}).get("kind")})

        # ---- 表结果 ---------------------------------------------------
        for tname, tv in (ev.get("tables") or {}).items():
            ctv = (cv.get("tables") or {}).get(tname) or {}
            for rng, rv in (tv.get("results") or {}).items():
                if not isinstance(rv, dict):
                    continue
                en_label = rv.get("name")
                cn_label = ((ctv.get("results") or {}).get(rng) or {}).get("name")
                brows = res_bind.get((packkey, ent, tname, rng)) or \
                        res_bind.get((packkey, None, tname, rng)) or []
                bound = [b for b in brows if b.get("documentUuid") or b.get("documentId")]
                tgt, tpk = None, None
                for b in bound:
                    uuid = b.get("documentUuid")
                    if uuid:
                        tp, tid = snb.parse_uuid(uuid, packkey)
                    else:
                        tp, tid = (b.get("documentCollection") or packkey), b.get("documentId")
                    t = snb.resolve(ids, tid, *(tp.split(".", 1) + [None])[:2]) if tid else None
                    if t:
                        tgt, tpk = t, tp
                        break
                rows.append({"kind": "result", "packkey": packkey, "pack": pv["file"],
                             "repo": os.path.basename(pv["repo"]),
                             "path": f"{ent}.tables.{tname}.results.{rng}.name",
                             "en": en_label, "cn": cn_label,
                             "bound": bool(bound), "has_rows": bool(brows),
                             "rtype": (brows[0].get("type") if brows else None),
                             "tgt_en": (tgt or {}).get("name"), "tgt_pk": tpk,
                             "tgt_kind": (tgt or {}).get("kind")})

print(f"rows={len(rows)}  notes={sum(1 for r in rows if r['kind']=='note')} "
      f"results={sum(1 for r in rows if r['kind']=='result')}")

# --------------------------------------------------------------- 全库英文名 -> 中文名
lib = collections.defaultdict(collections.Counter)
for pk, idx in indexes.items():
    for role, m in idx.items():
        for en_name, c in m.items():
            lib[(role, en_name)].update(c)

ROLE_SETS = {"note": ("pages", "journals"),
             "result": ("items", "entries", "tables", "actors")}


def lib_lookup(kind, en_name):
    out = collections.Counter()
    for role in ROLE_SETS[kind]:
        out.update(lib.get((role, (en_name or "").strip())) or {})
    return out


# --------------------------------------------------------------- A. NOT_BOUND 桶
print("\n=== A. NOT_BOUND：没有目标可点，但英文与某文档英文名逐字相同 ===")
a_hits = []
for r in rows:
    if r["bound"] or not r["en"]:
        continue
    cand = lib_lookup(r["kind"], r["en"])
    if not cand or not r["cn"]:
        continue
    if r["cn"] not in cand:
        a_hits.append(dict(r, target_cn=dict(cand)))
print(f"命中 {len(a_hits)}")
for h in a_hits:
    print(f"  [{h['kind']}] {h['pack']} :: {h['path']}\n"
          f"     EN {h['en']!r}  CN {h['cn']!r}  同名文档中文 {h['target_cn']}")

# --------------------------------------------------------------- B. BY_DESIGN 近似别名
print("\n=== B. BY_DESIGN：英文只差一点点（归一化后相同 / 互为子串），中文却不共享内核 ===")
b_hits = []
for r in rows:
    if not r["bound"] or not r["tgt_en"] or not r["en"]:
        continue
    if r["en"].strip() == r["tgt_en"].strip():
        continue                                   # 这是闸自己管的 OK/BROKEN 档
    en_a, en_b = norm(r["en"]), norm(r["tgt_en"])
    if not en_a or not en_b:
        continue
    related = (en_a == en_b) or (en_a in en_b) or (en_b in en_a)
    if not related:
        continue
    tgt_role = "pages" if r["kind"] == "note" else None
    cand = collections.Counter()
    if tgt_role:
        cand.update((indexes.get(r["tgt_pk"], {}) or {}).get("pages", {}).get(r["tgt_en"], {}))
        cand.update(lib_lookup("note", r["tgt_en"]))
    else:
        cand.update(lib_lookup("result", r["tgt_en"]))
    core = cn_core(r["cn"])
    shared = any(core and (core in cn_core(c) or cn_core(c) in core) for c in cand)
    b_hits.append(dict(r, target_cn=dict(cand), en_equal_norm=(en_a == en_b),
                       cn_shares_core=shared))
bad = [h for h in b_hits if not h["cn_shares_core"] and h["target_cn"] and h["cn"]]
print(f"英文近似 {len(b_hits)} 条，其中中文内核不共享 {len(bad)} 条")
for h in bad:
    print(f"  [{h['kind']}] {h['pack']} :: {h['path']}\n"
          f"     EN {h['en']!r} -> 目标 {h['tgt_en']!r}  (归一化相同={h['en_equal_norm']})\n"
          f"     CN {h['cn']!r} -> 目标 {h['target_cn']}")

# --------------------------------------------------------------- C. 孪生包一致性
print("\n=== C. 孪生包：英文逐字相同的同名针脚/结果行，中文是否一致 ===")
twin = collections.defaultdict(dict)
for r in rows:
    if not r["packkey"].startswith("ember."):
        continue
    if r["packkey"] not in ("ember.adventure", "ember.crucible-adventure"):
        continue
    key = (r["kind"], r["path"].split(".", 1)[1], r["en"])
    twin[key][r["packkey"]] = r
c_hits = []
for key, d in twin.items():
    if len(d) < 2:
        continue
    cns = {pk: v["cn"] for pk, v in d.items()}
    if len(set(cns.values())) > 1:
        c_hits.append({"key": list(key), "cn": cns,
                       "tgt": {pk: (v["tgt_en"], v["tgt_pk"]) for pk, v in d.items()}})
print(f"两包都有、英文相同、中文不同：{len(c_hits)}")
for h in c_hits:
    print(f"  {h['key'][0]} {h['key'][1]}\n     EN {h['key'][2]!r}\n     {h['cn']}\n     tgt {h['tgt']}")

# --------------------------------------------------------------- D. 跨包目标复查
print("\n=== D. 针脚目标在别的包里（闸只查当前包索引，可能藏 BROKEN） ===")
d_hits = []
for r in rows:
    if r["kind"] != "note" or not r["tgt_en"]:
        continue
    if r["en"].strip() != r["tgt_en"].strip():
        continue
    own = (indexes[r["packkey"]].get("pages", {}) or {}).get(r["tgt_en"], {})
    if own:
        continue                                    # 当前包里查得到，闸已覆盖
    cand = lib_lookup("note", r["tgt_en"])
    if cand and r["cn"] and r["cn"] not in cand:
        d_hits.append(dict(r, target_cn=dict(cand)))
print(f"当前包索引查不到、全库查得到且中文不同：{len(d_hits)}")
for h in d_hits:
    print(f"  {h['pack']} :: {h['path']}\n     EN {h['en']!r}  CN {h['cn']!r}  全库 {h['target_cn']}")

json.dump({"A_not_bound": a_hits, "B_alias": b_hits, "C_twin": c_hits, "D_crosspack": d_hits},
          open(os.path.join(SCRATCH, "g2_residual.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n-> g2_residual.json")
