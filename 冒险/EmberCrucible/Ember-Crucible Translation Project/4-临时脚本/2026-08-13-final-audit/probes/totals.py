# -*- coding: utf-8 -*-
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_validate as V
from enrich_inventory import walk_json

TOK = re.compile(r"@(Embed|embed|UUID)\[([^\]]*)\]", re.ASCII)


def flat(p):
    d = json.load(open(p, encoding="utf-8"))
    s = []
    walk_json(d, [], s)
    return dict(s)


tot = collections.Counter()
per = collections.defaultdict(collections.Counter)
for repo, base in V.REPOS.items():
    for side in ("en", "cn"):
        d = os.path.join(base, "compendium", side)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json") or fn == "_source.json":
                continue
            for jp, s in flat(os.path.join(d, fn)).items():
                for name, rx in V.PATTERNS:
                    n = len(rx.findall(s))
                    if n:
                        tot[(side, name)] += n
                        per[(repo, fn, side)][name] += n
                for m in TOK.finditer(s):
                    tot[(side, "core:@" + m.group(1))] += 1
names = sorted({k[1] for k in tot})
print("%-22s %8s %8s %6s" % ("enricher", "EN", "CN", "diff"))
te = tc = 0
for n in names:
    e, c = tot[("en", n)], tot[("cn", n)]
    te += e
    tc += c
    print("%-22s %8d %8d %6d" % (n, e, c, c - e))
print("%-22s %8d %8d %6d" % ("TOTAL", te, tc, tc - te))
