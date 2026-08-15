# -*- coding: utf-8 -*-
"""U3: resolve every @UUID target in findings[151:226] to its document name.

Chain: target last segment -> ember_ids.json (LevelDB dump: id -> EN name)
       -> name_index.json (EN doc name -> majority CN doc name).
The EN name is a fact from the packs; the CN name is the strongest basis for
the label (PROJECT.md decision ladder).
"""
import json, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SC = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-12-fix\reports"

d = json.load(open(SC + "/uuid_swap.json", encoding="utf-8"))
ids = json.load(open(os.path.join(R, "ember_ids.json"), encoding="utf-8"))
nidx = json.load(open(os.path.join(R, "name_index.json"), encoding="utf-8"))

lo, hi = int(sys.argv[1]) if len(sys.argv) > 1 else 151, int(sys.argv[2]) if len(sys.argv) > 2 else 226
rows = []
for i in range(lo, hi):
    x = d["findings"][i]
    key = x["key"]
    info = ids.get(key) or {}
    en_name = info.get("name")
    ni = nidx.get(en_name) if en_name else None
    rows.append({
        "i": i, "repo": x["repo"], "pack": x["pack"], "path": x["path"],
        "batch_path": x["batch_path"], "target": x["target"], "key": key,
        "en_label": x["en_label"], "cn_label": x["cn_label"],
        "majority": x["majority"]["label"], "msup": x["majority"]["support"],
        "mtot": x["majority"]["total"], "basis": x.get("basis"),
        "own_share": x.get("own_share"),
        "doc_en_name": en_name, "doc_type": info.get("type"), "via": info.get("via"),
        "doc_cn_name": ni["cn"] if ni else None,
        "doc_cn_n": (ni["n"], ni["total"]) if ni else None,
        "doc_cn_alts": ni.get("alts") if ni else None,
    })

json.dump(rows, open(SC + "/u3_resolved.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

for r in rows:
    print(f"{r['i']:3} en={r['en_label']!r:32} cn={r['cn_label']!r:26} "
          f"maj={r['majority']!r}({r['msup']}/{r['mtot']}) "
          f"| docEN={r['doc_en_name']!r} docCN={r['doc_cn_name']!r} {r['doc_cn_n']} "
          f"alts={r['doc_cn_alts']}")
print(f"\nunresolved ids: {sum(1 for r in rows if not r['doc_en_name'])}")
