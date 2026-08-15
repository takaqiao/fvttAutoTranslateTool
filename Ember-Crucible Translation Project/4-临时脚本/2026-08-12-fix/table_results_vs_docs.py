# -*- coding: utf-8 -*-
"""RollTable result names vs the document they point at.

Babele's `referencedDocumentField` prefers an explicit translation when one
exists, so a result row translated differently from the item document it draws
is visible at the table: the GM rolls 「刺剑」 and the player's pack holds
「细剑 Rapier」.

Gate is the English: a result row only counts when its English is byte-identical
to some item's English name in the same pack.
"""
import argparse, json, os, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def item_names(ev, cv):
    """{english item name: Counter(chinese name)} over pack items + actor items."""
    out = collections.defaultdict(collections.Counter)

    def add(en_items, cn_items):
        for k, v in (en_items or {}).items():
            enn = v.get("name") if isinstance(v, dict) else None
            cnn = ((cn_items or {}).get(k) or {}).get("name") if isinstance(cn_items, dict) else None
            if isinstance(enn, str) and isinstance(cnn, str) and cnn:
                out[enn.strip()][cnn] += 1

    add(ev.get("items"), cv.get("items"))
    for an, av in (ev.get("actors") or {}).items():
        cav = (cv.get("actors") or {}).get(an) or {}
        add(av.get("items"), cav.get("items"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    en = json.load(open(os.path.join(a.repo, "compendium", "en", a.pack), encoding="utf-8"))
    cn = json.load(open(os.path.join(a.repo, "compendium", "cn", a.pack), encoding="utf-8"))
    rows, total, matched = [], 0, 0
    for adv, ev in en.get("entries", {}).items():
        cv = cn.get("entries", {}).get(adv) or {}
        docs = item_names(ev, cv)
        for tname, tv in (ev.get("tables") or {}).items():
            ctv = (cv.get("tables") or {}).get(tname) or {}
            for rk, rv in (tv.get("results") or {}).items():
                enn = rv.get("name") if isinstance(rv, dict) else None
                if not isinstance(enn, str):
                    continue
                total += 1
                cnn = ((ctv.get("results") or {}).get(rk) or {}).get("name")
                cand = docs.get(enn.strip())
                if not cand:
                    continue
                matched += 1
                best = cand.most_common(1)[0][0]
                if cnn != best and not (cnn and cnn in cand):
                    rows.append({"batch_path": f"{adv}.tables.{tname}.results.{rk}.name",
                                 "en": enn, "row_cn": cnn, "doc_cn": best,
                                 "doc_variants": dict(cand)})
    print(f"results total={total}  matched-an-item-name={matched}  MISMATCH={len(rows)}")
    for r in rows:
        print(f"  {r['batch_path']}\n     EN  {r['en']!r}\n     row {r['row_cn']!r}\n     doc {r['doc_cn']!r} {r['doc_variants']}")
    if a.out:
        json.dump({r["batch_path"]: r["doc_cn"] for r in rows},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> batch {a.out}")


if __name__ == "__main__":
    main()
