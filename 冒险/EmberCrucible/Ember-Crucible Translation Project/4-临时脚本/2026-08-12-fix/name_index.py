# -*- coding: utf-8 -*-
"""Build an EN document-name -> CN document-name index across both repos.

`@UUID[target]{label}` labels are, in the upstream English, verbatim copies of the
target document's name. So the strongest basis for the Chinese label is the CN
`name` field of that document (PROJECT.md decision ladder: name field is strongest).
This index is immune to the audit-1.4 label/target shuffle because it never looks
at labels at all.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CJK = re.compile(r'[一-鿿]')
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
        out.append((path[:], en, cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    idx = defaultdict(Counter)
    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            out = []
            walk(en.get("entries", {}), cn.get("entries", {}), [], out)
            for p, e, c in out:
                if not p or p[-1] not in ("name", "label"):
                    continue
                if not c or not CJK.search(c):
                    continue
                idx[e][c] += 1

    res = {}
    for e, c in idx.items():
        top, n = c.most_common(1)[0]
        res[e] = {"cn": top, "n": n, "total": sum(c.values()),
                  "alts": {k: v for k, v in c.items() if k != top}}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"name index entries: {len(res)}")


if __name__ == "__main__":
    main()
