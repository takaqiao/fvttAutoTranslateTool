"""H2-A: compare the extracted-English *coverage* of our crucible extractor
against Padhiver/Crucible-FR's independent extractor.

Both sides extracted from crucible 0.10.1 packs with independent tooling, so a
string one side captured and the other did not points at a real coverage
difference. Unlike the 2026-08-06 crosscheck (top-level field names of shared
entries only) this walks the whole tree, normalises collection keys (they key
embedded collections by _id, we key by name / array index), and diffs the
*value multiset* per (entry, path-shape) so nested gaps such as
Scene.levels[].name become visible.

Usage:
  python h2_pathset.py [--json OUT.json]
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

OURS = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\en"
THEIRS = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-fr\compendium\en"

ID16 = re.compile(r"^[A-Za-z0-9]{16}$")
# parents whose direct children are collection members, not field names
COLLECTIONS = {
    "actions", "effects", "results", "levels", "notes", "regions", "behaviors",
    "deltaTokens", "tokens", "journals", "scenes", "macros", "tables", "items",
    "pages", "folders", "categories", "outcomes", "changes", "actors",
}


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def walk(node, path, out):
    """Collect (normalised shape, concrete path, string value) for every leaf."""
    if isinstance(node, dict):
        last = path[-1][0] if path else None
        collection = last in COLLECTIONS
        for k, v in node.items():
            key = "*" if (collection or ID16.match(k)) else k
            walk(v, path + [(key, k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [("*", str(i))], out)
    elif isinstance(node, str):
        if node.strip():
            out.append((".".join(s for s, _ in path),
                        ".".join(c for _, c in path), node))


def entries_by_name(pack):
    out = {}
    for k, v in pack.get("entries", {}).items():
        if isinstance(v, dict) and isinstance(v.get("name"), str):
            out.setdefault(v["name"], v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--max-ex", type=int, default=12)
    args = ap.parse_args()

    report = {}
    files = sorted(f for f in os.listdir(OURS)
                   if f.endswith(".json") and not f.startswith("_"))
    for fn in files:
        tp = os.path.join(THEIRS, fn)
        if not os.path.exists(tp):
            report[fn] = {"error": "not in Crucible-FR"}
            continue
        na = entries_by_name(load(os.path.join(OURS, fn)))
        nb = entries_by_name(load(tp))
        shared = sorted(set(na) & set(nb))

        sa, sb = Counter(), Counter()          # shape -> leaf count
        miss = defaultdict(list)               # shape -> [(entry, concrete, val)]
        extra = defaultdict(list)
        n_miss = Counter()
        n_extra = Counter()
        for name in shared:
            la, lb = [], []
            walk(na[name], [], la)
            walk(nb[name], [], lb)
            sa.update(s for s, _, _ in la)
            sb.update(s for s, _, _ in lb)
            va, vb = defaultdict(Counter), defaultdict(Counter)
            pa, pb = defaultdict(dict), defaultdict(dict)
            for s, c, t in la:
                va[s][t] += 1
                pa[s].setdefault(t, c)
            for s, c, t in lb:
                vb[s][t] += 1
                pb[s].setdefault(t, c)
            for s in set(va) | set(vb):
                d = vb[s] - va[s]          # theirs has, we don't
                for t, n in d.items():
                    n_miss[s] += n
                    if len(miss[s]) < args.max_ex:
                        miss[s].append([name, pb[s][t], t[:200]])
                d2 = va[s] - vb[s]         # we have, theirs doesn't
                for t, n in d2.items():
                    n_extra[s] += n
                    if len(extra[s]) < args.max_ex:
                        extra[s].append([name, pa[s][t], t[:200]])

        report[fn] = {
            "entries_ours": len(na), "entries_theirs": len(nb),
            "shared": len(shared),
            "names_only_ours": sorted(set(na) - set(nb))[:30],
            "names_only_theirs": sorted(set(nb) - set(na))[:30],
            "leaves_ours": sum(sa.values()), "leaves_theirs": sum(sb.values()),
            "shapes_only_theirs": sorted(s for s in sb if s not in sa),
            "shapes_only_ours": sorted(s for s in sa if s not in sb),
            "missing_by_shape": dict(n_miss.most_common()),
            "extra_by_shape": dict(n_extra.most_common()),
            "missing_examples": {k: v for k, v in miss.items()},
            "extra_examples": {k: v for k, v in extra.items()},
        }

    for fn, d in report.items():
        if "error" in d:
            print(f"-- {fn}: {d['error']}")
            continue
        print(f"\n== {fn}  entries {d['entries_ours']}/{d['entries_theirs']} "
              f"shared={d['shared']}  leaves {d['leaves_ours']}/{d['leaves_theirs']}")
        if d["names_only_theirs"]:
            print(f"   entry names only theirs: {d['names_only_theirs']}")
        if d["names_only_ours"]:
            print(f"   entry names only ours  : {d['names_only_ours']}")
        if d["shapes_only_theirs"]:
            print(f"   SHAPES ONLY THEIRS: {d['shapes_only_theirs']}")
        if d["shapes_only_ours"]:
            print(f"   shapes only ours  : {d['shapes_only_ours']}")
        if d["missing_by_shape"]:
            print(f"   WE MISS (shape -> n): {d['missing_by_shape']}")
            for s, ex in d["missing_examples"].items():
                print(f"     [{s}]")
                for e in ex[:6]:
                    print(f"        {e[0]} | {e[1]} | {e[2][:110]}")
        if d["extra_by_shape"]:
            print(f"   we have extra (shape -> n): {d['extra_by_shape']}")
            for s, ex in d["extra_examples"].items():
                print(f"     [{s}]")
                for e in ex[:4]:
                    print(f"        {e[0]} | {e[1]} | {e[2][:110]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
