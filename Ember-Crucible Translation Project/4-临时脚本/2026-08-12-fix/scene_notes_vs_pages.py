# -*- coding: utf-8 -*-
"""Map-pin labels (`scenes.*.notes`) vs the journal page they open.

A note whose English is byte-identical to some journal page name is, in
practice, a pin that opens that page. If the Chinese pin says one thing and the
Chinese page title says another, the GM clicks 「A」 and a window titled 「B」
opens. No existing gate looks at this: both sides are valid Chinese, both have
100% coverage, and neither contains markup.
"""
import argparse, json, os, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def page_names(en_root, cn_root):
    """{english page name: Counter(chinese page name)} over all journals."""
    out = collections.defaultdict(collections.Counter)
    for ename, ev in (en_root.get("journals") or {}).items():
        cv = (cn_root.get("journals") or {}).get(ename) or {}
        for pname, pv in (ev.get("pages") or {}).items():
            cp = (cv.get("pages") or {}).get(pname) or {}
            cnn = cp.get("name")
            if isinstance(cnn, str) and cnn:
                out[pname][cnn] += 1
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
        pages = page_names(ev, cv)
        for sname, sv in (ev.get("scenes") or {}).items():
            csv_ = (cv.get("scenes") or {}).get(sname) or {}
            for nk, nv in (sv.get("notes") or {}).items():
                total += 1
                cnote = (csv_.get("notes") or {}).get(nk)
                if not isinstance(nv, str):
                    continue
                cand = pages.get(nv.strip())
                if not cand:
                    continue
                matched += 1
                best = cand.most_common(1)[0][0]
                if cnote != best and not (cnote and cnote in cand):
                    rows.append({
                        "batch_path": f"{adv}.scenes.{sname}.notes.{nk}",
                        "en": nv, "pin_cn": cnote, "page_cn": best,
                        "page_variants": dict(cand)})
    print(f"notes total={total}  matched-a-page-name={matched}  MISMATCH={len(rows)}")
    for r in rows:
        print(f"  {r['batch_path']}\n     EN   {r['en']!r}\n     pin  {r['pin_cn']!r}\n     page {r['page_cn']!r}  {r['page_variants']}")
    if a.out:
        json.dump({r["batch_path"]: r["page_cn"] for r in rows},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> batch {a.out}")


if __name__ == "__main__":
    main()
