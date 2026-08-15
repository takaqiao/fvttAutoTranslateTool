# -*- coding: utf-8 -*-
"""Every (EN label -> CN label) pair the library currently uses for a given target id.

Reads the WORKING TREE (post audit-3 apply), pairs EN and CN occurrences positionally
inside each leaf, so an EN label that legitimately differs from the document name is
visible instead of being averaged away by a bare majority vote.
"""
import argparse, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
SKIP = {"_id", "path", "_variants", "_when"}
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?')


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))


def occ(s, key):
    r = []
    for m in MARK.finditer(s or ""):
        tgt = (m.group(2) or "").split()[0].split("#")[0]
        if tgt.split(".")[-1] == key:
            r.append(m.group(3))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", required=True)
    a = ap.parse_args()
    for key in a.key:
        print("#" * 96)
        print("KEY", key)
        pairs = Counter()
        for repo in REPOS:
            ed = os.path.join(ROOT, repo, "compendium", "en")
            cd = os.path.join(ROOT, repo, "compendium", "cn")
            for fn in sorted(os.listdir(ed)):
                if not fn.endswith(".json") or fn.startswith("_"):
                    continue
                en = json.load(open(os.path.join(ed, fn), encoding="utf-8"))
                cp = os.path.join(cd, fn)
                cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
                rows = []
                walk(en.get("entries", {}), cn.get("entries", {}), [], rows)
                for p, e, c in rows:
                    if key not in e:
                        continue
                    el, cl = occ(e, key), occ(c, key)
                    if len(el) != len(cl):
                        print(f"  !! count mismatch {fn} :: {p}  EN={el} CN={cl}")
                        continue
                    for A, B in zip(el, cl):
                        pairs[(A, B)] += 1
                        print(f"  {fn[:32]:34} {p[:74]:76} {A!r} -> {B!r}")
        print("  ---- pair census ----")
        for (A, B), n in pairs.most_common():
            print(f"   {n:3}  {A!r} -> {B!r}")


main()
