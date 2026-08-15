# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_hits.json")
d = json.load(open(p, encoding="utf-8"))
for r in d["rows"]:
    if "Control Water" in r["path"] or "slinerak" in r["path"]:
        print("="*100)
        print(r["pack"], "|", r["path"])
        print("--- EN ---")
        print(r["en"])
        print("--- CN ---")
        print(r["cn"])
