# -*- coding: utf-8 -*-
"""Dump en->cn pairs for one normalised field-path class (see field_census.py).

Read-only.  Usage:
  python class_dump.py "<class-glob>" [--repo ember|crucible|both] [--limit N]

`<class-glob>` is matched against the normalised path (fnmatch).
"""
import json, os, sys, fnmatch, argparse, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {"ember": os.path.join(ROOT, "1-Ember汉化插件"),
         "crucible": os.path.join(ROOT, "2-Crucible汉化插件")}
COLLECTION_KEYS = {
    "entries", "folders", "levels", "tokens", "categories", "drawings", "notes",
    "journals", "scenes", "macros", "playlists", "tables", "items", "actors",
    "pages", "sounds", "results", "effects", "regions", "behaviors",
    "outcomes", "changes", "actions",
}


def walk(node, path, norm, out, collapse=False):
    if isinstance(node, dict):
        for k, v in node.items():
            if collapse:
                walk(v, path + [k], norm + ["*"], out, False)
            else:
                walk(v, path + [k], norm + [k], out, k in COLLECTION_KEYS)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], norm + ["[]"], out, False)
    elif isinstance(node, str):
        out[".".join(path)] = (".".join(norm), node)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cls")
    ap.add_argument("--repo", default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uniq", action="store_true", help="collapse identical en->cn pairs")
    ap.add_argument("--out")
    a = ap.parse_args()

    repos = REPOS if a.repo == "both" else {a.repo: REPOS[a.repo]}
    rows = []
    for repo, base in repos.items():
        endir = os.path.join(base, "compendium", "en")
        cndir = os.path.join(base, "compendium", "cn")
        names = sorted(set(os.listdir(endir)) | set(os.listdir(cndir)))
        for fn in names:
            if not fn.endswith(".json") or fn == "_source.json":
                continue
            en, cn = {}, {}
            ep, cp = os.path.join(endir, fn), os.path.join(cndir, fn)
            if os.path.exists(ep):
                walk(json.load(open(ep, encoding="utf-8")), [], [], en)
            if os.path.exists(cp):
                walk(json.load(open(cp, encoding="utf-8")), [], [], cn)
            keys = list(en.keys()) + [k for k in cn if k not in en]
            for k in keys:
                n = (en.get(k) or cn.get(k))[0]
                if not fnmatch.fnmatch(n, a.cls):
                    continue
                rows.append({"repo": repo, "pack": fn, "path": k, "cls": n,
                             "en": en.get(k, (None, None))[1],
                             "cn": cn.get(k, (None, None))[1]})
    if a.uniq:
        seen, uq = set(), []
        for r in rows:
            key = (r["cls"], r["en"], r["cn"])
            if key in seen:
                continue
            seen.add(key)
            uq.append(r)
        rows = uq
    print(f"# {len(rows)} rows for class {a.cls}")
    for r in (rows[:a.limit] if a.limit else rows):
        print(f'{r["repo"][:1]}|{r["pack"][:34]:34s}|{r["cls"][-46:]:46s}| EN={r["en"]!r} CN={r["cn"]!r}')
    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
