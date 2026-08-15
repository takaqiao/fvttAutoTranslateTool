# -*- coding: utf-8 -*-
"""Align <strong> bodies between EN and CN leaf by leaf.

Gives an English-gated majority table: for English <strong>X</strong>, which
Chinese body sits at the same index. Leaves whose <strong> counts differ are
skipped (alignment would be bogus).
"""
import argparse, json, os, re, sys, collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SKIP = {"_id", "path", "_variants", "_when"}
RX = re.compile(r'<strong>(.*?)</strong>', re.S)


def walk(en, cn, p, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, p + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, p + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(p), en, cn if isinstance(cn, str) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--en", help="only report this English body (exact)")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out")
    a = ap.parse_args()

    table = collections.defaultdict(collections.Counter)
    where = collections.defaultdict(list)
    for repo in a.repo:
        ed = os.path.join(repo, "compendium", "en")
        cd = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(ed)):
            if not fn.endswith(".json"):
                continue
            en = json.load(open(os.path.join(ed, fn), encoding="utf-8"))
            cp = os.path.join(cd, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            rows = []
            walk(en.get("entries", {}), cn.get("entries", {}), [], rows)
            for p, e, c in rows:
                if not c:
                    continue
                E, C = RX.findall(e), RX.findall(c)
                if len(E) != len(C) or not E:
                    continue
                for x, y in zip(E, C):
                    table[x.strip()][y.strip()] += 1
                    if x.strip() == y.strip():
                        where[x.strip()].append((os.path.basename(repo), fn, p))

    items = [(k, v) for k, v in table.items() if re.search(r'[A-Za-z]{2}', k)]
    items.sort(key=lambda kv: -kv[1].get(kv[0], 0))
    for k, v in items[: a.top]:
        untl = v.get(k, 0)
        if a.en and k != a.en:
            continue
        if not untl and not a.en:
            continue
        print(f"EN <strong>{k}</strong>   untranslated={untl}   renderings={v.most_common(6)}")
    if a.out:
        json.dump({k: dict(v) for k, v in table.items()},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump({k: v for k, v in where.items()},
                  open(a.out.replace(".json", ".where.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("->", a.out)


if __name__ == "__main__":
    main()
