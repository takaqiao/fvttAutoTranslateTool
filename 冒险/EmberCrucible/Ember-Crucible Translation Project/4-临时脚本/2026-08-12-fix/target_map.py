# -*- coding: utf-8 -*-
"""(target, EN label) -> majority CN label used for that same pair elsewhere.

English-gated, per PROJECT.md's "先查英文再落笔": a CN label only votes for a
target when the EN leaf uses the SAME target with the SAME English label.  Pairing
inside a leaf is by target, so it survives prose reordering; targets that occur
more than once in a leaf are skipped (ambiguous).
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CJK = re.compile(r'[一-鿿]')
SKIP_KEYS = {"_id", "path", "_variants", "_when"}
MARK = re.compile(r'(@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])(\{([^{}]*)\})?')


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))


def by_target(s):
    d = defaultdict(list)
    for m in MARK.finditer(s):
        d[m.group(1)].append(m.group(3))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    votes = defaultdict(Counter)
    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            rows = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)
            for p, e, c in rows:
                if not c:
                    continue
                eb, cb = by_target(e), by_target(c)
                for t, els in eb.items():
                    cls = cb.get(t)
                    if len(els) != 1 or not cls or len(cls) != 1:
                        continue          # ambiguous inside this leaf
                    el, cl = els[0], cls[0]
                    if el is None or cl is None or not CJK.search(cl):
                        continue
                    votes[t + "\t" + el][cl] += 1
    res = {}
    for k, c in votes.items():
        top, n = c.most_common(1)[0]
        res[k] = {"cn": top, "n": n, "total": sum(c.values()),
                  "alts": {x: y for x, y in c.items() if x != top}}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("target+enlabel keys:", len(res))


if __name__ == "__main__":
    main()
