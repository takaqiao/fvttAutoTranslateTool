# -*- coding: utf-8 -*-
"""For a target id, list every leaf in both repos that links to it, with EN and CN labels.

Positional alignment per leaf (same target, k-th occurrence) so a target that carries two
different English labels in one leaf is not mis-attributed.
"""
import argparse, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?')
SKIP = {"_id", "path", "_variants", "_when"}

def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP: continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))

def labels(s, key):
    res = []
    for m in MARK.finditer(s or ""):
        t = (m.group(2) or "").split()[0].split("#")[0] if m.group(2) else ""
        if t.split(".")[-1] == key:
            res.append(m.group(3))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--cn-only", action="store_true", help="also scan CN-only leaves")
    a = ap.parse_args()
    pairs = Counter()
    for repo in REPOS:
        ed = os.path.join(ROOT, repo, "compendium", "en")
        cd = os.path.join(ROOT, repo, "compendium", "cn")
        for fn in sorted(os.listdir(ed)):
            if not fn.endswith(".json") or fn.startswith("_"): continue
            en = json.load(open(os.path.join(ed, fn), encoding="utf-8"))
            cp = os.path.join(cd, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            rows = []
            walk(en, cn, [], rows)
            for p, e, c in rows:
                el = labels(e, a.key); cl = labels(c, a.key) if c else []
                if not el and not cl: continue
                print(f"{repo[:1]} {fn} :: {p}")
                if len(el) == len(cl):
                    for x, y in zip(el, cl):
                        print(f"      {x!r:45} -> {y!r}")
                        pairs[(x, y)] += 1
                else:
                    print(f"      EN={el} CN={cl}   *** count mismatch ***")
                    pairs[("<mismatch>", str(cl))] += 1
    print("\n=== (EN label, CN label) leaf counts ===")
    for (x, y), n in pairs.most_common():
        print(f"{n:4}  {x!r:45} -> {y!r}")
main()
