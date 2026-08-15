# -*- coding: utf-8 -*-
import json, sys, re, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
inv = json.load(open(os.path.join(HERE, "inventory.json"), encoding="utf-8"))

for key, items in inv.items():
    at = collections.Counter()
    bb = collections.Counter()
    for it in items:
        if it["kind"] == "at":
            at[it["name"]] += 1
        else:
            a = it["args"].strip()
            m = re.match(r"^(/?[A-Za-z][A-Za-z0-9_]*)", a)
            bb[m.group(1) if m else "<%s>" % a[:20]] += 1
    print("=== " + key)
    print("  @NAME:", dict(at.most_common()))
    print("  [[cmd:", dict(bb.most_common(40)))
