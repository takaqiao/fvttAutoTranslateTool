# -*- coding: utf-8 -*-
"""Census of every translated field-path shape in the two plugins' compendium JSON.

Read-only. Emits, per repo and per side (en/cn), the set of normalised field
paths with leaf counts, so we can ask "which judge has ever looked at this
path class?".

Normalisation: dict levels that are *collections* (keyed by entry id / name)
collapse to `*`. Collection keys are taken from mappings.mjs converters.
"""
import json, os, sys, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {
    "ember": os.path.join(ROOT, "1-Ember汉化插件"),
    "crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
}

# keys whose VALUE is a collection object (keyed by id/name) -> collapse one level
COLLECTION_KEYS = {
    "entries", "folders", "levels", "tokens", "categories", "drawings", "notes",
    "journals", "scenes", "macros", "playlists", "tables", "items", "actors",
    "pages", "sounds", "results", "effects", "regions", "behaviors",
    "outcomes", "changes", "actions",
}


def walk(node, path, out, collapse=False):
    if isinstance(node, dict):
        for k, v in node.items():
            if collapse:
                walk(v, path + ["*"], out, collapse=False)
            else:
                nxt = path + [k]
                walk(v, nxt, out, collapse=(k in COLLECTION_KEYS))
    elif isinstance(node, list):
        for v in node:
            walk(v, path + ["[]"], out, collapse=False)
    else:
        p = ".".join(path)
        out[p][0] += 1
        if isinstance(node, str):
            out[p][1] += len(node)
        else:
            out[p][2] += 1  # non-string leaf


def main():
    grand = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    perfile = {}
    for repo, base in REPOS.items():
        for side in ("en", "cn"):
            d = os.path.join(base, "compendium", side)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".json"):
                    continue
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    data = json.load(f)
                out = collections.defaultdict(lambda: [0, 0, 0])
                walk(data, [], out, collapse=False)
                perfile[(repo, side, fn)] = dict(out)
                for p, c in out.items():
                    g = grand[(repo, side)][p]
                    g[0] += c[0]; g[1] += c[1]; g[2] += c[2]
    res = {
        "grand": {f"{r}/{s}": {p: v for p, v in sorted(d.items())} for (r, s), d in grand.items()},
        "perfile": {f"{r}/{s}/{fn}": {p: v for p, v in sorted(d.items())} for (r, s, fn), d in perfile.items()},
    }
    outp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "field_census.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for key in sorted(res["grand"]):
        print("=" * 20, key)
        for p, v in sorted(res["grand"][key].items(), key=lambda kv: -kv[1][0]):
            print(f"  {v[0]:7d} leaves  {v[1]:9d} chars  nonstr={v[2]:5d}  {p}")


if __name__ == "__main__":
    main()
