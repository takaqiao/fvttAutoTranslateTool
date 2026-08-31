# -*- coding: utf-8 -*-
"""Which EN keys carrying enricher tokens have no CN counterpart at all?"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_validate as V
from enrich_inventory import walk_json

TOK = re.compile(r"@(Embed|embed|UUID)\[([^\]]*)\]|\[\[/\w+", re.ASCII)
CJK = re.compile(r"[一-鿿]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def flat(p):
    d = json.load(open(p, encoding="utf-8"))
    s = []
    walk_json(d, [], s)
    return dict(s)


rows = []
for repo, base in V.REPOS.items():
    en = os.path.join(base, "compendium", "en")
    cn = os.path.join(base, "compendium", "cn")
    for fn in sorted(os.listdir(en)):
        if not fn.endswith(".json") or fn == "_source.json":
            continue
        if not os.path.isfile(os.path.join(cn, fn)):
            continue
        E, C = flat(os.path.join(en, fn)), flat(os.path.join(cn, fn))
        for jp, ev in E.items():
            if jp in C:
                continue
            n = len(TOK.findall(ev))
            if not n:
                continue
            # strip tags + enricher tokens, see if any prose survives
            bare = re.sub(r"<[^>]+>", "", ev)
            bare = re.sub(r"@\w+\[[^\]]*\](\{[^}]*\})?", "", bare)
            bare = re.sub(r"\[\[[^\]]*\]\](\{[^}]*\})?", "", bare)
            words = LATIN_WORD.findall(bare)
            rows.append((repo, fn, jp, n, len(words), bare.strip()[:80]))
print("EN keys with enricher tokens and NO cn key:", len(rows),
      " tokens:", sum(r[3] for r in rows))
for r in sorted(rows, key=lambda x: -x[4]):
    print("  %-32s %-70s tok=%d prosewords=%d | %s" % (r[1], r[2][:70], r[3], r[4], r[5]))
