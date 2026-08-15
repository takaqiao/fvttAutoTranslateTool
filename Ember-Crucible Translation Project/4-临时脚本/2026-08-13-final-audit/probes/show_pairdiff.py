# -*- coding: utf-8 -*-
import json, os, sys, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
rep = json.load(open(os.path.join(HERE, "pairdiff.json"), encoding="utf-8"))
want = sys.argv[1] if len(sys.argv) > 1 else "PARAM_DIFF"
filt = sys.argv[2] if len(sys.argv) > 2 else None
n = 0
for r in rep:
    if r["type"] != want:
        continue
    blob = " ".join(r["en_only"] + r["cn_only"])
    if filt and filt not in blob:
        continue
    n += 1
    print("--- [%s] %s :: %s" % (r["repo"], r["file"], r["jpath"]))
    for s in r["en_only"]:
        print("   EN-only: " + s[:300])
    for s in r["cn_only"]:
        print("   CN-only: " + s[:300])
print("total", n)
