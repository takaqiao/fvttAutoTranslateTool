# -*- coding: utf-8 -*-
"""Dump aligned EN/CN leaf-string pairs from a Babele translation repo.

Path convention matches qa/validate_translations.py ('.'.join(path)).

Usage examples:
  python pair_dump.py --repo "<repoDir>" --pack crucible.rules.json --n 40 --seed 7
  python pair_dump.py --repo "<repoDir>" --pack ember.crucible-adventure.json --slice 1/4 --n 60
  python pair_dump.py --repo "<repoDir>" --pack all --grep-en "Electricity" --n 200 --full
  python pair_dump.py --repo "<repoDir>" --pack all --stats
"""
import argparse, json, os, random, re, sys, io

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append({"path": p,
                    # key form accepted by qa/apply_translations.py batches:
                    # relative to `entries`, i.e. without the leading "entries."
                    "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                    "en": en,
                    "cn": cn if isinstance(cn, str) else None})


def collect(repo, packs):
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    rows = []
    for fn in packs:
        ep = os.path.join(en_dir, fn)
        if not os.path.isfile(ep):
            continue
        en = json.load(open(ep, encoding="utf-8"))
        cp = os.path.join(cn_dir, fn)
        cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
        sub = []
        walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
        for r in sub:
            r["pack"] = fn
        rows.extend(sub)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", default="all", help="filename, comma list, or 'all'")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slice", help="i/k  -> take deterministic shard i of k")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=0)
    ap.add_argument("--grep-en", help="regex on the English side")
    ap.add_argument("--grep-cn", help="regex on the Chinese side")
    ap.add_argument("--grep-path", help="regex on the path")
    ap.add_argument("--missing-only", action="store_true", help="only rows with no Chinese")
    ap.add_argument("--full", action="store_true", help="do not truncate values")
    ap.add_argument("--trunc", type=int, default=1200)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, "compendium", "en")
    if a.pack == "all":
        packs = sorted(f for f in os.listdir(en_dir)
                       if f.endswith(".json") and not f.startswith("_"))
    else:
        packs = [p.strip() for p in a.pack.split(",") if p.strip()]

    rows = collect(a.repo, packs)

    if a.stats:
        agg = {}
        for r in rows:
            d = agg.setdefault(r["pack"], {"leaves": 0, "cn": 0, "en_chars": 0})
            d["leaves"] += 1
            d["en_chars"] += len(r["en"])
            if r["cn"]:
                d["cn"] += 1
        for k, v in sorted(agg.items()):
            pct = 100.0 * v["cn"] / v["leaves"] if v["leaves"] else 0
            print(f"{k:<42} leaves={v['leaves']:<7} withCN={v['cn']:<7} {pct:5.1f}%  enChars={v['en_chars']}")
        print(f"TOTAL leaves={len(rows)}")
        return

    if a.missing_only:
        rows = [r for r in rows if not r["cn"]]
    if a.min_chars:
        rows = [r for r in rows if len(r["en"]) >= a.min_chars]
    if a.max_chars:
        rows = [r for r in rows if len(r["en"]) <= a.max_chars]
    if a.grep_en:
        rx = re.compile(a.grep_en)
        rows = [r for r in rows if rx.search(r["en"])]
    if a.grep_cn:
        rx = re.compile(a.grep_cn)
        rows = [r for r in rows if r["cn"] and rx.search(r["cn"])]
    if a.grep_path:
        rx = re.compile(a.grep_path)
        rows = [r for r in rows if rx.search(r["path"])]

    total = len(rows)
    if a.slice:
        i, k = (int(x) for x in a.slice.split("/"))
        rows = [r for j, r in enumerate(sorted(rows, key=lambda r: (r["pack"], r["path"])))
                if j % k == (i - 1) % k]

    rnd = random.Random(a.seed)
    if a.n and len(rows) > a.n:
        rows = rnd.sample(rows, a.n)
    rows.sort(key=lambda r: (r["pack"], r["path"]))

    if not a.full:
        for r in rows:
            if len(r["en"]) > a.trunc:
                r["en"] = r["en"][: a.trunc] + f"…<+{len(r['en'])-a.trunc}>"
            if r["cn"] and len(r["cn"]) > a.trunc:
                r["cn"] = r["cn"][: a.trunc] + f"…<+{len(r['cn'])-a.trunc}>"

    payload = {"matched": total, "returned": len(rows), "rows": rows}
    txt = json.dumps(payload, ensure_ascii=False, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
        print(f"matched={total} returned={len(rows)} -> {a.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
