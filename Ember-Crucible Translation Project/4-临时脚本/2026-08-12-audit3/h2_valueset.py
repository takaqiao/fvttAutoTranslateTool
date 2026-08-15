"""H2-A2: per-pack GLOBAL string-set diff, ours vs Crucible-FR.

The per-shape diff (h2_pathset.py) over-reports whenever the two extractors key
a collection differently (they use _id, we collapse identical duplicates onto a
name key). What actually matters for coverage is whether a given English string
exists ANYWHERE in the pack file — that is what Babele will look up. So this
compares the raw set of string leaves.

Usage: python h2_valueset.py [--json OUT.json]
"""
import argparse
import json
import os
from collections import defaultdict

OURS = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\en"
THEIRS = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-fr\compendium\en"


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out.setdefault(node, []).append(".".join(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    report = {}
    for fn in sorted(f for f in os.listdir(OURS)
                     if f.endswith(".json") and not f.startswith("_")):
        tp = os.path.join(THEIRS, fn)
        if not os.path.exists(tp):
            continue
        a, b = {}, {}
        leaves(load(os.path.join(OURS, fn)).get("entries", {}), [], a)
        leaves(load(tp).get("entries", {}), [], b)
        only_b = sorted(set(b) - set(a))
        only_a = sorted(set(a) - set(b))
        report[fn] = {
            "n_strings_ours": len(a), "n_strings_theirs": len(b),
            "only_theirs": [[s[:220], b[s][0]] for s in only_b],
            "only_ours": [[s[:220], a[s][0]] for s in only_a],
        }
        print(f"== {fn}  distinct strings {len(a)}/{len(b)}  "
              f"ONLY-THEIRS={len(only_b)}  only-ours={len(only_a)}")
        for s in only_b[:40]:
            print(f"   THEIRS-ONLY | {b[s][0]} | {s[:150]}")
        for s in only_a[:20]:
            print(f"   ours-only   | {a[s][0]} | {s[:150]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
