# -*- coding: utf-8 -*-
"""Group untranslated labels by English text and attach evidence for review.

Evidence ladder (PROJECT.md 第 8 节 / 依据阶梯):
  1. target document's CN `name` field, with the bilingual English tail stripped
     -- measured 12539:0 in-library that {labels} use the bare Chinese part
  2. majority CN label used for the same English label elsewhere
"""
import argparse, json, re, sys, html
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CJK = re.compile(r'[一-鿿]')


def strip_tail(cnname, en):
    """'远古月神殿 Ancient Lunar Shrine' + en='Ancient Lunar Shrine' -> '远古月神殿'."""
    for cand in {en, html.unescape(en)}:
        t = " " + cand
        if cnname.endswith(t) and CJK.search(cnname[: -len(t)]):
            return cnname[: -len(t)].strip()
    # tail is latin-ish and the head has CJK
    m = re.match(r"^(.*?[一-鿿])\s+([A-Za-z0-9'’&;#.,:!?()\-À-ɏ ]{2,})$", cnname)
    if m and not CJK.search(m.group(2)):
        return m.group(1).strip()
    return cnname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="append", required=True)
    ap.add_argument("--name-index", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--out")
    ap.add_argument("--only-new", action="store_true")
    ap.add_argument("--only-auto", action="store_true")
    a = ap.parse_args()

    ni = json.load(open(a.name_index, encoding="utf-8"))
    lm = json.load(open(a.label_map, encoding="utf-8"))
    groups = defaultdict(lambda: {"n": 0, "ctx": [], "paths": [], "targets": Counter()})
    for pf in a.plan:
        for it in json.load(open(pf, encoding="utf-8")):
            for l in it["labels"]:
                g = groups[l["en"] or ""]
                g["n"] += 1
                g["targets"][l["target"]] += 1
                if len(g["ctx"]) < 3:
                    g["ctx"].append(l["ctx"])
                    g["paths"].append(it["batch_path"])

    rows = []
    for en, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        r = {"en": en, "n": g["n"], "targets": len(g["targets"])}
        if en in ni:
            r["name"] = ni[en]["cn"]
            r["name_stripped"] = strip_tail(ni[en]["cn"], en)
            r["name_n"] = f'{ni[en]["n"]}/{ni[en]["total"]}'
            if ni[en]["alts"]:
                r["name_alts"] = ni[en]["alts"]
        if en in lm:
            r["lbl"] = lm[en]["cn"]
            r["lbl_n"] = f'{lm[en]["n"]}/{lm[en]["total"]}'
            if lm[en]["alts"]:
                r["lbl_alts"] = lm[en]["alts"]
        r["ctx"] = g["ctx"]
        r["paths"] = g["paths"]
        rows.append(r)

    if a.only_new:
        rows = [r for r in rows if "name" not in r and "lbl" not in r]
    if a.only_auto:
        rows = [r for r in rows if "name" in r or "lbl" in r]
    txt = json.dumps(rows, ensure_ascii=False, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
        print(f"groups={len(rows)} labels={sum(r['n'] for r in rows)} -> {a.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
