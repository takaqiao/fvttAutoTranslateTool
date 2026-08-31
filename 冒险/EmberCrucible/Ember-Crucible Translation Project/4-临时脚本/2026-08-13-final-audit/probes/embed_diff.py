# -*- coding: utf-8 -*-
"""Probe: @Embed / @UUID / @Advantage / @CriticalX token multiset comparison EN vs CN,
per identical JSON key path. Uses a flat non-nesting regex (these enrichers never contain
a `]` inside their argument list).
"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_validate as V
from enrich_inventory import walk_json

TOK = re.compile(r"@(Embed|embed|UUID|Advantage|CriticalSuccess|CriticalFailure|Loot|Scroll)\[([^\]]*)\]", re.ASCII)


def flat(path):
    d = json.load(open(path, encoding="utf-8"))
    sink = []
    walk_json(d, [], sink)
    return dict(sink)


def sig(s):
    return collections.Counter("@%s[%s]" % (m.group(1), m.group(2)) for m in TOK.finditer(s))


def main():
    rows = []
    tot = collections.Counter()
    for repo, base in V.REPOS.items():
        endir = os.path.join(base, "compendium", "en")
        cndir = os.path.join(base, "compendium", "cn")
        for fn in sorted(os.listdir(endir)):
            if not fn.endswith(".json") or fn == "_source.json":
                continue
            cp = os.path.join(cndir, fn)
            if not os.path.isfile(cp):
                continue
            en, cn = flat(os.path.join(endir, fn)), flat(cp)
            for jp, ev in en.items():
                es = sig(ev)
                cv = cn.get(jp)
                if cv is None:
                    if es:
                        tot["missing_key"] += sum(es.values())
                    continue
                cs = sig(cv)
                oe, oc = es - cs, cs - es
                if oe or oc:
                    rows.append({"repo": repo, "file": fn, "jpath": jp,
                                 "en_only": sorted(oe.elements()),
                                 "cn_only": sorted(oc.elements()),
                                 "en_len": len(ev), "cn_len": len(cv)})
                    for t in oe.elements():
                        tot["EN_only_" + t.split("[")[0]] += 1
                    for t in oc.elements():
                        tot["CN_only_" + t.split("[")[0]] += 1
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(rows, open(os.path.join(here, "embeddiff.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(tot)
    print("locations:", len(rows))
    for r in rows[:40]:
        print("--- %s | %s | %s  (len en=%d cn=%d)" % (r["repo"], r["file"], r["jpath"], r["en_len"], r["cn_len"]))
        for x in r["en_only"][:8]:
            print("    EN-only:", x[:150])
        for x in r["cn_only"][:8]:
            print("    CN-only:", x[:150])


if __name__ == "__main__":
    main()
