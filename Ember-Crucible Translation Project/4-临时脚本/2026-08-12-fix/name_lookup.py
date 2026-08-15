# -*- coding: utf-8 -*-
"""Find leaves whose ENGLISH equals (or matches) a term, print the CN side.

Used to answer "what does the library call X?" with the English gate on.
"""
import argparse, json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SKIP_KEYS = {"_id", "path", "_variants", "_when"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--en", required=True, help="regex the ENGLISH leaf must match")
    ap.add_argument("--exact", action="store_true", help="EN must equal --en literally")
    ap.add_argument("--max-en", type=int, default=90, help="only leaves with EN shorter than this")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--census", help="comma list of CN candidate strings to count")
    a = ap.parse_args()
    rx = re.compile(a.en)
    rows = []
    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for p, e, c in sub:
                rows.append((os.path.basename(repo), fn, p, e, c))
    if a.exact:
        hits = [r for r in rows if r[3].strip() == a.en]
    else:
        hits = [r for r in rows if rx.search(r[3]) and len(r[3]) <= a.max_en]
    print(f"EN-matching short leaves: {len(hits)}")
    cnt = collections.Counter(r[4] for r in hits)
    for v, n in cnt.most_common(a.show):
        ex = next(r for r in hits if r[4] == v)
        print(f"  {n:>4}  {v!r}   e.g. {ex[1]}::{ex[2]}   EN={ex[3]!r}")
    if a.census:
        allhits = [r for r in rows if rx.search(r[3])]
        print(f"\ncensus over ALL {len(allhits)} EN-matching leaves:")
        for t in a.census.split(","):
            print(f"  {t}: {sum(1 for r in allhits if t in (r[4] or ''))}")


if __name__ == "__main__":
    main()
