"""H2-A: deep field-path coverage cross-check, ours vs Padhiver/Crucible-FR.

The 2026-08-06 crosscheck only compared TOP-LEVEL field names per entry.
This one walks every leaf and compares *canonical* leaf paths, so nested gaps
(system.*, actions.*, pages.*, effects.*) become visible.

Key problem solved here: the two extractors key nested documents differently
(we key `pages`/`journals`/`actors` by the document's `name`, Crucible-FR keys
them by `_id`).  A raw path diff is therefore ~100% noise.  Canonicalisation:
for every dict child that is itself a document (has a string `name`), the path
segment becomes that `name`, on both sides.  Array indices become `[]`.

Both sides were extracted from crucible 0.10.1 packs by independent tooling,
so a surviving disagreement points at a real bug in one of them.

Usage:
  python h2_fr_field_coverage.py [--json out.json] [--pack crucible.rules.json]
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

OURS_EN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\en"
FR_EN = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-fr\compendium\en"

TAG = re.compile(r"<[^>]+>")


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def plain(s):
    return TAG.sub(" ", s)


def canon_seg(key, val):
    """Path segment that is stable across the two extractors' keying schemes."""
    if isinstance(val, dict) and isinstance(val.get("name"), str) and val["name"].strip():
        return val["name"]
    return key


def leaves(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            seg = canon_seg(k, v)
            leaves(v, f"{prefix}.{seg}" if prefix else seg, out)
    elif isinstance(obj, list):
        for v in obj:
            leaves(v, f"{prefix}[]", out)
    elif isinstance(obj, str) and obj.strip():
        # arrays collapse to one key; keep every value
        out.setdefault(prefix, []).append(obj)
    return out


def shape(path):
    """Strip the concrete doc-name segments so paths can be histogrammed.

    Keeps only segments that look like schema field names (lowercase-ish,
    no spaces).  Anything else becomes '*'.
    """
    parts = []
    for s in path.replace("[]", "").split("."):
        parts.append(s if re.fullmatch(r"[a-z_][A-Za-z0-9_]*", s) else "*")
    return ".".join(parts)


def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        elif isinstance(v, list) and isinstance(dst.get(k), list):
            for x in v:
                if x not in dst[k]:
                    dst[k].append(x)
        else:
            dst.setdefault(k, v)
    return dst


def by_name(entries):
    """Re-key by the entry's own `name`, MERGING same-name entries.

    crucible.pregens holds two Actor documents per hero (Level 1 / Level 6);
    Crucible-FR keys them separately by _id, this project merges them under the
    name key (PROJECT.md section 8, 2026-08-06 "同名文档若内容可合并则合并到名字键").
    Comparing only the first FR document of a name manufactures ~9k characters
    of phantom "only ours" — merge instead.
    """
    out = {}
    for k, v in entries.items():
        if isinstance(v, dict) and isinstance(v.get("name"), str):
            if v["name"] in out:
                deep_merge(out[v["name"]][1], json.loads(json.dumps(v)))
            else:
                out[v["name"]] = (k, json.loads(json.dumps(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--pack")
    args = ap.parse_args()

    report = {"packs": {}, "totals": {}}
    files = sorted(f for f in os.listdir(OURS_EN)
                   if f.endswith(".json") and not f.startswith("_"))
    if args.pack:
        files = [args.pack]

    print(f"{'pack':<34}{'ours':>6}{'fr':>6}{'shared':>7}{'oOnly':>6}{'fOnly':>6}"
          f"{'ourLeaf':>8}{'frLeaf':>7}{'MISS':>6}{'MISSch':>8}{'EXTRA':>6}{'EXTRAch':>8}")
    T = Counter()

    for fn in files:
        fp = os.path.join(FR_EN, fn)
        if not os.path.exists(fp):
            print(f"{fn:<34}  --- not in Crucible-FR ---")
            report["packs"][fn] = {"status": "absent_in_fr"}
            continue
        a, b = load(os.path.join(OURS_EN, fn)), load(fp)
        ae, be = a.get("entries", {}), b.get("entries", {})
        na, nb = by_name(ae), by_name(be)
        shared = sorted(set(na) & set(nb))

        our_shapes, fr_shapes = Counter(), Counter()
        miss = defaultdict(lambda: {"n": 0, "chars": 0, "examples": []})
        extra = defaultdict(lambda: {"n": 0, "chars": 0, "examples": []})
        nl_a = nl_b = 0
        for name in shared:
            la = leaves(na[name][1])
            lb = leaves(nb[name][1])
            nl_a += sum(len(v) for v in la.values())
            nl_b += sum(len(v) for v in lb.values())
            for p in la:
                our_shapes[shape(p)] += len(la[p])
            for p in lb:
                fr_shapes[shape(p)] += len(lb[p])
            for p in set(lb) - set(la):
                d = miss[shape(p)]
                d["n"] += 1
                d["chars"] += sum(len(plain(s)) for s in lb[p])
                if len(d["examples"]) < 8:
                    d["examples"].append({"entry": name, "path": p,
                                          "fr_en": lb[p][0][:260]})
            for p in set(la) - set(lb):
                d = extra[shape(p)]
                d["n"] += 1
                d["chars"] += sum(len(plain(s)) for s in la[p])
                if len(d["examples"]) < 8:
                    d["examples"].append({"entry": name, "path": p,
                                          "our_en": la[p][0][:260]})

        mc = sum(v["chars"] for v in miss.values())
        ec = sum(v["chars"] for v in extra.values())
        mn = sum(v["n"] for v in miss.values())
        en = sum(v["n"] for v in extra.values())
        print(f"{fn:<34}{len(ae):>6}{len(be):>6}{len(shared):>7}"
              f"{len(set(na)-set(nb)):>6}{len(set(nb)-set(na)):>6}"
              f"{nl_a:>8}{nl_b:>7}{mn:>6}{mc:>8}{en:>6}{ec:>8}")
        T["miss"] += mn
        T["miss_chars"] += mc
        T["extra"] += en
        T["extra_chars"] += ec

        report["packs"][fn] = {
            "our_entries": len(ae), "fr_entries": len(be), "shared": len(shared),
            "only_ours_entries": sorted(set(na) - set(nb)),
            "only_fr_entries": sorted(set(nb) - set(na)),
            "our_leaves": nl_a, "fr_leaves": nl_b,
            "missing_by_shape": {k: v for k, v in
                                 sorted(miss.items(), key=lambda kv: -kv[1]["chars"])},
            "extra_by_shape": {k: v for k, v in
                               sorted(extra.items(), key=lambda kv: -kv[1]["chars"])},
        }

    report["totals"] = dict(T)
    print(f"\nTOTAL  we-miss {T['miss']} leaves / {T['miss_chars']} chars   "
          f"only-ours {T['extra']} leaves / {T['extra_chars']} chars")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
