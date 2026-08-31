# -*- coding: utf-8 -*-
"""Merge evidence for every untranslated label and emit a decision worksheet.

Evidence, strongest first (PROJECT.md 依据阶梯: name 字段 > 同卷已译页 > 全库多数):
  T  target_map  -- CN label used for the SAME @UUID target elsewhere
  N  name_index  -- CN `name` of the document whose EN name equals the EN label
  L  label_map   -- majority CN label for the same EN label text

Auto rows are those where the available evidence agrees (after stripping the
bilingual English tail from name-index values).  Everything else is listed for
hand review.
"""
import argparse, json, re, sys, html
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CJK = re.compile(r'[一-鿿]')


def strip_tail(cnname, en):
    for cand in {en, html.unescape(en or "")}:
        if not cand:
            continue
        t = " " + cand
        if cnname.endswith(t) and CJK.search(cnname[: -len(t)]):
            return cnname[: -len(t)].strip()
    m = re.match(r"^(.*?[一-鿿])\s+([A-Za-z0-9'’&;#.,:!?()\-À-ɏ ]{2,})$", cnname)
    if m and not CJK.search(m.group(2)):
        return m.group(1).strip()
    return cnname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--name-index", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--target-map", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ni = json.load(open(a.name_index, encoding="utf-8"))
    lm = json.load(open(a.label_map, encoding="utf-8"))
    tm = json.load(open(a.target_map, encoding="utf-8"))

    rows = []
    for it in json.load(open(a.plan, encoding="utf-8")):
        for l in it["labels"]:
            en = l["en"]
            ev = {}
            tk = l["target"] + "\t" + (en or "")
            if tk in tm:
                ev["T"] = (tm[tk]["cn"], tm[tk]["n"], tm[tk]["total"])
            if en and en in ni:
                ev["N"] = (strip_tail(ni[en]["cn"], en), ni[en]["n"], ni[en]["total"])
            if en and en in lm:
                ev["L"] = (lm[en]["cn"], lm[en]["n"], lm[en]["total"])
            vals = {k: v[0] for k, v in ev.items()}
            agree = len(set(vals.values())) == 1 and vals
            rows.append({"path": it["batch_path"], "idx": l["idx"], "target": l["target"],
                         "en": en, "cn": l["cn"], "ev": ev,
                         "auto": (list(vals.values())[0] if agree else None),
                         "ctx": l["ctx"]})
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    auto = sum(1 for r in rows if r["auto"])
    print(f"labels={len(rows)}  agreed={auto}  needs-review={len(rows)-auto}")


if __name__ == "__main__":
    main()
