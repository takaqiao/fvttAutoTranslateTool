# -*- coding: utf-8 -*-
import json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
b = json.load(open(os.path.join(HERE, "badvalue.json"), encoding="utf-8"))
u = json.load(open(os.path.join(HERE, "unmatched.json"), encoding="utf-8"))

print("###### BADVALUE distinct tokens (side=en only, per file)")
agg = collections.defaultdict(collections.Counter)
for x in b:
    if x["side"] != "en":
        continue
    tok = re.match(r"^(\[\[/\w+ [^\]]*\]\]|@\w+\[[^\]]*\])", x["snip"])
    agg[(x["enricher"], x["why"])][ (tok.group(1) if tok else x["snip"][:60]), x["file"] ] += 1
for k in sorted(agg):
    print("--", k)
    for (t, f), n in agg[k].most_common(200):
        print("     %-52s %-32s %d" % (t[:52], f, n))

print()
print("###### UNMATCHED distinct tokens (side=en only)")
agg2 = collections.defaultdict(collections.Counter)
for x in u:
    if x["side"] != "en":
        continue
    agg2[x["head"]][(x["snip"][:70], x["file"])] += 1
for k in sorted(agg2):
    print("--", k)
    for (t, f), n in agg2[k].most_common(50):
        print("     %-70s %-32s %d" % (t, f, n))
