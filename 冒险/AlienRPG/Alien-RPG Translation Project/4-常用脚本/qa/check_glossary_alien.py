# -*- coding: utf-8 -*-
"""Structural validator for glossary_alien.{json,provenance,disputes,pending}.

Everything here is an INVARIANT, not a style preference.  Each one exists because
breaking it silently produces a wrong release rather than a visible error:

  flat shape          consumers read glossary_alien.json as {str: str} and nothing else
  no drift            glossary values must be exactly provenance.terms[<en>].cn
  authoritative count every _meta count must be recomputed from the data, never typed
  tier present        the owner's naming decision is per-entry; an entry with no tier
                      cannot be applied
  T-FROZEN identity   a frozen value that is not the English string breaks a lookup
  T-EXACT byte-equal  a skill-stunt Item name that is not byte-equal to the lang value
                      stops matching
  bilingual contract  '中文 English', ONE ASCII space, no parentheses, and the English
                      is the NAME, not the disambiguated glossary key
  pending disjoint    a pending term that also sits in the glossary defeats the point
                      of pending: the translator stops being asked
  dispute anchored    a dispute whose provisional value has drifted from the glossary
                      is a lie about what will ship

Run:  python check_glossary_alien.py
Exit: 0 clean, 1 on any violation.
"""
import json, io, os, sys, collections

G = r"C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project/7-其他内容/glossary"
LANG = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang"
TIERS = {"T-FROZEN", "T-EXACT", "T-BILINGUAL", "T-PLAIN"}


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = load(os.path.join(G, "glossary_alien.json"))
    prov = load(os.path.join(G, "glossary_alien.provenance.json"))
    disp = load(os.path.join(G, "glossary_alien.disputes.json"))
    pend = load(os.path.join(G, "glossary_alien.pending.json"))
    cn = load(os.path.join(LANG, "cn.json"))["ALIENRPG"]
    p, m = prov["terms"], prov["_meta"]
    bad = []

    def ck(cond, msg):
        if not cond:
            bad.append(msg)

    ck(isinstance(g, dict), "glossary is not an object")
    ck(all(isinstance(k, str) and isinstance(v, str) for k, v in g.items()),
       "glossary is not flat {str: str}")
    ck(set(g) == set(p), "glossary/provenance key sets differ: %r" %
       (sorted(set(g) ^ set(p))[:8],))
    for k in set(g) & set(p):
        ck(g[k] == p[k]["cn"], "value drift on %r: %r vs %r" % (k, g[k], p[k]["cn"]))

    ck(m["count"] == len(p), "_meta.count %r != %d" % (m.get("count"), len(p)))
    ck(m["tier_counts"] == dict(sorted(collections.Counter(
        v["tier"] for v in p.values()).items())), "_meta.tier_counts stale")
    ck(m["category_counts"] == dict(sorted(collections.Counter(
        v["category"] for v in p.values()).items())), "_meta.category_counts stale")
    ck(disp["_meta"]["count"] == len(disp["disputes"]), "disputes _meta.count stale")
    ck(pend["_meta"]["count"] == len(pend["terms"]), "pending _meta.count stale")

    skillvals = {cn[k] for k in cn if k.startswith("Skill") and isinstance(cn[k], str)}
    for k, v in sorted(p.items()):
        for f in ("cn", "tier", "category", "why_this_won", "candidates"):
            ck(f in v, "%r: missing %s" % (k, f))
        ck(v.get("tier") in TIERS, "%r: bad tier %r" % (k, v.get("tier")))
        ck(bool(v.get("candidates")), "%r: empty candidates" % k)
        for c in v.get("candidates", []):
            for f in ("zh", "source", "count", "evidence"):
                ck(f in c, "%r: candidate missing %s" % (k, f))
        t = v.get("tier")
        if t == "T-FROZEN":
            ck(v["cn"].upper() == k.upper(),
               "T-FROZEN must be the English string: %r -> %r" % (k, v["cn"]))
        if t == "T-EXACT":
            ck(v["cn"] in skillvals,
               "T-EXACT not byte-equal to any cn.json ALIENRPG.Skill* value: %r -> %r" % (k, v["cn"]))
        if t == "T-BILINGUAL":
            b = v.get("bilingual_name")
            base = k.split(" (")[0] if (" (" in k and k.endswith(")")) else k
            ck(b == v["cn"] + " " + base,
               "bilingual_name contract broken: %r -> %r (want %r)" % (k, b, v["cn"] + " " + base))
        else:
            ck("bilingual_name" not in v, "%r: bilingual_name on a %s entry" % (k, t))

    over = sorted(set(pend["terms"]) & set(g))
    ck(not over, "pending terms also present in the glossary: %r" % (over,))

    for did, d in sorted(disp["disputes"].items()):
        ck(d["en"] in g, "%s: en %r absent from the glossary" % (did, d["en"]))
        if d["en"] in g:
            ck(g[d["en"]] == d["provisional_cn"],
               "%s: provisional %r != shipped %r" % (did, d["provisional_cn"], g[d["en"]]))
        ck(len(d.get("sides", [])) >= 2, "%s: fewer than two sides" % did)
        for s in d.get("sides", []):
            for f in ("cn", "argument", "sources", "gated_count", "evidence"):
                ck(f in s, "%s: side missing %s" % (did, f))

    for k, v in sorted(pend["terms"].items()):
        for f in ("kind", "candidates", "corpus", "why_unresolved", "next_step"):
            ck(f in v, "pending %r: missing %s" % (k, f))

    print("terms %d | tiers %s" % (len(p), m["tier_counts"]))
    print("disputes %d | pending %d | collisions %d" %
          (len(disp["disputes"]), len(pend["terms"]),
           len(m["chinese_value_collisions"]["collisions"])))
    if bad:
        print("\n%d VIOLATION(S):" % len(bad))
        for b in bad:
            print("  !", b)
        return 1
    print("\nALL INVARIANTS HOLD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
