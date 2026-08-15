# -*- coding: utf-8 -*-
"""Per-location EN vs CN comparison using the STRICT upstream patterns.

For each (file, jpath) present on both sides, compare the multiset of
(enricher_name, argument_groups) tuples. Labels are excluded (they are translated).
Also compare the multiset of *unmatched* system-enricher candidates.
"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_validate as V
from enrich_inventory import walk_json

LABEL_GROUP = {   # index of the group that is a display label, to be ignored
    "hazard": 1, "rule": 1, "ref": 1, "scroll": 1, "loot": 2,
}


def flat(path):
    d = json.load(open(path, encoding="utf-8"))
    sink = []
    walk_json(d, [], sink)
    return dict(sink)


def strict_sig(s):
    out = collections.Counter()
    for name, rx in V.PATTERNS:
        for mm in rx.finditer(s):
            g = list(mm.groups())
            li = LABEL_GROUP.get(name)
            if li is not None and li < len(g):
                g[li] = None
            out["%s%r" % (name, tuple(g))] += 1
    return out


def loose_sig(s):
    out = collections.Counter()
    for st, head, mt, snip in V.analyse(s):
        if mt is None and V.classify(head) == "sys":
            out["UNMATCHED:" + head] += 1
    return out


def main():
    rows = []
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
                cv = cn.get(jp)
                es = strict_sig(ev) + loose_sig(ev)
                if cv is None:
                    if es:
                        rows.append({"t": "MISSING_CN", "repo": repo, "file": fn, "jpath": jp,
                                     "en_only": sorted(es.elements()), "cn_only": []})
                    continue
                cs = strict_sig(cv) + loose_sig(cv)
                oe, oc = es - cs, cs - es
                if oe or oc:
                    rows.append({"t": "DIFF", "repo": repo, "file": fn, "jpath": jp,
                                 "en_only": sorted(oe.elements()), "cn_only": sorted(oc.elements())})
            # CN keys absent from EN
            for jp, cv in cn.items():
                if jp in en:
                    continue
                cs = strict_sig(cv) + loose_sig(cv)
                if cs:
                    rows.append({"t": "EXTRA_CN", "repo": repo, "file": fn, "jpath": jp,
                                 "en_only": [], "cn_only": sorted(cs.elements())})
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(rows, open(os.path.join(here, "sidediff.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(collections.Counter(r["t"] for r in rows))
    for r in rows:
        if r["t"] != "DIFF":
            continue
        print("--- %s | %s | %s" % (r["repo"], r["file"], r["jpath"]))
        for x in r["en_only"]:
            print("    EN-only:", x)
        for x in r["cn_only"]:
            print("    CN-only:", x)


if __name__ == "__main__":
    main()
