# -*- coding: utf-8 -*-
"""U3: one line per finding — EN label, CN label, library majority, and the LIVE
CN `name` of the target document (all spellings, with counts)."""
import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
R = os.path.join(P, r"4-临时脚本\2026-08-12-fix\reports")
SC = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"

d = json.load(open(SC + "/uuid_swap.json", encoding="utf-8"))
ids = json.load(open(os.path.join(R, "ember_ids.json"), encoding="utf-8"))
live = json.load(open(SC + "/u3_name_live.json", encoding="utf-8"))

lo, hi = int(sys.argv[1]), int(sys.argv[2])
out = []
for i in range(lo, hi):
    x = d["findings"][i]
    info = ids.get(x["key"]) or {}
    en_name = info.get("name")
    hits = live.get(en_name or "", [])
    c = Counter(h["cn"] for h in hits if h["cn"])
    out.append({**{k: x[k] for k in ("repo", "pack", "path", "batch_path", "target",
                                     "key", "en_label", "cn_label", "i", "basis")},
                "majority": x["majority"], "own_share": x.get("own_share"),
                "doc_en": en_name, "doc_type": info.get("type"),
                "doc_cn": c.most_common(), "doc_hits": len(hits)})
    m = x["majority"]
    print(f"{i:3} en={str(x['en_label'])[:28]:28} cn={x['cn_label'][:16]:16} "
          f"maj={m['label'][:14]:14}{m['support']}/{m['total']} "
          f"docEN={str(en_name)[:26]:26} docCN={c.most_common()}")
json.dump(out, open(SC + f"/u3_table_{lo}_{hi}.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
