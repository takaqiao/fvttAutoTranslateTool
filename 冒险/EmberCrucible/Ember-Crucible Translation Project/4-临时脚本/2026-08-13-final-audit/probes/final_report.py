# -*- coding: utf-8 -*-
"""Consolidated evidence dump: crucible-visible enricher failures in the crucible-side
packs, with EN/CN comparison. Drops the `@ref` namespace check (v2 false positive:
@ref[path] is an arbitrary property path on options.relativeTo, crucible-compiled.mjs:46973).
"""
import json, os, collections, sys
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "validate2.json"), encoding="utf-8"))

CRUCIBLE_PACKS = lambda fn: fn.startswith("crucible.") or (
    fn.startswith("ember.crucible-") or fn == "ember.crucible-adventure.json")

sel = [r for r in rows
       if CRUCIBLE_PACKS(r["file"])
       and r["swap"] != "dnd5e"
       and not (r.get("why", "").startswith("ref namespace"))]

for side in ("en", "cn"):
    c = collections.Counter((r["kind"], r.get("why") or r.get("head"), r["file"])
                            for r in sel if r["side"] == side)
    print("=== side=%s  total=%d" % (side, sum(c.values())))
    for k, v in sorted(c.items()):
        print("   %-9s %-34s %-32s %d" % (k[0], k[1], k[2], v))

print()
print("=== CN evidence sample (one location per distinct problem)")
seen = set()
for r in sel:
    if r["side"] != "cn":
        continue
    k = (r["kind"], r.get("why") or r.get("head"))
    if k in seen:
        continue
    seen.add(k)
    print("* %s | %s\n    %s\n    %s" % (k[0], k[1], r["jpath"], r["snip"][:110]))
