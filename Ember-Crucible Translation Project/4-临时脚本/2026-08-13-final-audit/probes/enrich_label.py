# -*- coding: utf-8 -*-
"""Probe: enrichers whose upstream pattern has NO trailing {label} capture, but which are
followed by a `{...}` in the content. Those braces are NOT consumed by the enricher and
render as literal text next to the enriched element.

Also compares EN vs CN so that a CN-introduced label is separable from an upstream one.
"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_validate as V
from enrich_inventory import walk_json

# Enrichers whose upstream regex ends WITHOUT (?:{([^}]+)})?
NO_LABEL = {"award", "counterspell", "skillCheck", "knowledge", "language", "dnd5eSkill",
            "talent", "condition", "action", "spell", "milestone",
            "date", "ancestry", "culture", "path", "attunement", "emberLanguage",
            "soundscape", "eventState", "outcome", "advantage", "critical",
            "emberKnowledge"}
PAT = {n: rx for n, rx in V.PATTERNS}


def flat(path):
    d = json.load(open(path, encoding="utf-8"))
    sink = []
    walk_json(d, [], sink)
    return dict(sink)


def find(s):
    out = collections.Counter()
    detail = []
    for name in NO_LABEL:
        rx = PAT.get(name)
        if not rx:
            continue
        for mm in rx.finditer(s):
            if mm.end() < len(s) and s[mm.end()] == "{":
                out["%s|%s" % (name, mm.group(0))] += 1
                detail.append((name, s[mm.start():mm.end() + 40]))
    return out, detail


def main():
    rows = []
    for repo, base in V.REPOS.items():
        for side in ("en", "cn"):
            d = os.path.join(base, "compendium", side)
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".json") or fn == "_source.json":
                    continue
                for jp, s in flat(os.path.join(d, fn)).items():
                    c, det = find(s)
                    for name, snip in det:
                        rows.append({"repo": repo, "side": side, "file": fn, "jpath": jp,
                                     "enricher": name, "snip": snip})
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(rows, open(os.path.join(here, "labelbug.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    c = collections.Counter((r["repo"], r["side"], r["enricher"]) for r in rows)
    for k, v in sorted(c.items()):
        print(k, v)
    print("--- samples")
    seen = set()
    for r in rows:
        k = (r["repo"], r["side"], r["enricher"])
        if k in seen:
            continue
        seen.add(k)
        print(k, "|", r["file"], "|", r["snip"][:120])


if __name__ == "__main__":
    main()
