# -*- coding: utf-8 -*-
"""Probe: align EN and CN babele dicts by identical JSON key path, then compare the
*multiset of enricher argument strings* found in each side's string value.

Rationale: enricher arguments (skill ids, condition ids, event ids, UUIDs) must be
byte-identical between EN and CN. Any difference means the translator touched a
machine-readable parameter.

False-positive modes (documented):
  - A CN string may legitimately reorder enrichers in a sentence -> we compare
    multisets, so reordering is NOT flagged.
  - A CN value may be missing entirely (untranslated) -> reported separately as
    MISSING_CN, not as a param defect.
  - @UUID display labels {..} are excluded from comparison (labels are translated).
"""
import json, os, re, sys, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPOS = {
    "ember": os.path.join(ROOT, "1-Ember汉化插件"),
    "crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_inventory import _scan_at, _scan_bb, walk_json


def flat(path):
    d = json.load(open(path, encoding="utf-8"))
    sink = []
    walk_json(d, [], sink)
    return dict(sink)


def sigs(s):
    """Argument-only signatures (labels dropped)."""
    out = []
    for name, args, label, err in _scan_at(s):
        out.append("@%s[%s]" % (name, args))
    for inner, label, err in _scan_bb(s):
        out.append("[[%s]]" % inner)
    return collections.Counter(out)


def main():
    report = []
    for repo, base in REPOS.items():
        endir = os.path.join(base, "compendium", "en")
        cndir = os.path.join(base, "compendium", "cn")
        for fn in sorted(os.listdir(endir)):
            if not fn.endswith(".json") or fn == "_source.json":
                continue
            cp = os.path.join(cndir, fn)
            if not os.path.isfile(cp):
                continue
            en = flat(os.path.join(endir, fn))
            cn = flat(cp)
            for jp, ev in en.items():
                if not ("@" in ev or "[[" in ev):
                    continue
                es = sigs(ev)
                if not es:
                    continue
                cv = cn.get(jp)
                if cv is None:
                    report.append({"repo": repo, "file": fn, "jpath": jp,
                                   "type": "MISSING_CN", "en_only": sorted(es),
                                   "cn_only": []})
                    continue
                cs = sigs(cv)
                only_en = es - cs
                only_cn = cs - es
                if only_en or only_cn:
                    report.append({"repo": repo, "file": fn, "jpath": jp,
                                   "type": "PARAM_DIFF",
                                   "en_only": sorted(only_en.elements()),
                                   "cn_only": sorted(only_cn.elements())})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairdiff.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    byt = collections.Counter(r["type"] for r in report)
    print(byt)
    # summarize the distinct enricher heads involved
    heads = collections.Counter()
    for r in report:
        if r["type"] != "PARAM_DIFF":
            continue
        for s in r["en_only"] + r["cn_only"]:
            m = re.match(r"^(@\w+|\[\[/?\w+)", s)
            heads[m.group(1) if m else s[:14]] += 1
    print(heads.most_common(40))


if __name__ == "__main__":
    main()
