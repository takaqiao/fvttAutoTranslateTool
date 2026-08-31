# -*- coding: utf-8 -*-
"""Content-scale audit of the ember English baseline: documents, leaves, chars,
and how much of the total is duplicated across the D&D5e / Crucible twin
adventure packs."""
import json, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leafdiff import leaves, load

def scoped(d):
    return {k: v for k, v in leaves(d).items() if k.startswith("entries.")}

def main(d):
    tot_leaf = tot_char = 0
    per = {}
    allvals = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        lv = scoped(load(os.path.join(d, f)))
        c = sum(len(v) for v in lv.values())
        per[f] = (len(lv), c)
        tot_leaf += len(lv); tot_char += c
        allvals[f] = lv
    print(f"{'pack':<38}{'leaves':>8}{'chars':>12}")
    for f, (n, c) in per.items():
        print(f"{f:<38}{n:>8}{c:>12,}")
    print(f"{'TOTAL':<38}{tot_leaf:>8}{tot_char:>12,}")
    # twin overlap
    a = allvals.get("ember.adventure.json", {})
    b = allvals.get("ember.crucible-adventure.json", {})
    if a and b:
        sa, sb = set(a.values()), set(b.values())
        print(f"\ntwin packs: dnd5e={len(a)} leaves / {len(sa)} distinct strings; "
              f"crucible={len(b)} leaves / {len(sb)} distinct strings")
        print(f"  distinct strings shared by both: {len(sa & sb)}")
        print(f"  union of distinct strings      : {len(sa | sb)}")
        print(f"  chars of that union            : {sum(len(s) for s in (sa|sb)):,}")
    # whole-baseline dedupe
    every = {}
    for f, lv in allvals.items():
        for k, v in lv.items():
            every.setdefault(v, 0)
            every[v] += 1
    print(f"\nwhole baseline: {tot_leaf} leaves -> {len(every)} distinct strings, "
          f"{sum(len(s) for s in every):,} chars once-each")

if __name__ == "__main__":
    main(sys.argv[1])
