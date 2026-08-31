# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_hits.json")
d = json.load(open(p, encoding="utf-8"))
RX = re.compile(r"(the\s+)?(next|following)\s+round", re.I)
seen = set()
for r in d["rows"]:
    key = (r["path"], r["en"][:60])
    if r["pack"] == "ember.adventure.json":
        continue  # twin duplicate of crucible-adventure; identical content
    print("="*95)
    print(r["pack"], "|", r["path"])
    for m in RX.finditer(r["en"]):
        print("  EN ...", r["en"][max(0,m.start()-170):m.end()+90].replace("\n"," "))
    cn = r["cn"] or ""
    for m in re.finditer(r"轮|回合", cn):
        print("  CN ...", cn[max(0,m.start()-40):m.start()+25].replace("\n"," "))
