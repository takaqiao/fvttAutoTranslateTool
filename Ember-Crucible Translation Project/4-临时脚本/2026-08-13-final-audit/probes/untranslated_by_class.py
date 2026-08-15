# -*- coding: utf-8 -*-
"""Per normalised field class: how many CN leaves carry no CJK at all
(i.e. still pure English / identical to EN), and how many are missing.

This is the check that `scan_en_residue.py` (>=5 English words) and
`scan_content_coverage.py` (>=120 English chars) both structurally cannot make
for short leaves such as folder names, level names, note pins, table rows.
Read-only.
"""
import json, os, re, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {"ember": os.path.join(ROOT, "1-Ember汉化插件"),
         "crucible": os.path.join(ROOT, "2-Crucible汉化插件")}
COLLECTION_KEYS = {
    "entries", "folders", "levels", "tokens", "categories", "drawings", "notes",
    "journals", "scenes", "macros", "playlists", "tables", "items", "actors",
    "pages", "sounds", "results", "effects", "regions", "behaviors",
    "outcomes", "changes", "actions",
}
CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
LATIN = re.compile(r"[A-Za-z]")


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


rows = collections.defaultdict(lambda: {"n": 0, "same": 0, "nocjk": 0, "missing": 0,
                                        "ex_same": [], "ex_missing": []})
for repo, base in REPOS.items():
    endir = os.path.join(base, "compendium", "en")
    cndir = os.path.join(base, "compendium", "cn")
    for fn in sorted(os.listdir(endir)):
        if not fn.endswith(".json") or fn == "_source.json":
            continue
        cp = os.path.join(cndir, fn)
        en, cn = {}, {}
        walk(json.load(open(os.path.join(endir, fn), encoding="utf-8")), [], [], en)
        if os.path.exists(cp):
            walk(json.load(open(cp, encoding="utf-8")), [], [], cn)
        for k, (norm, ev) in en.items():
            if norm.startswith("packs") or norm in ("extractedAt", "extractedBy",
                                                    "mappingTarget", "packageId",
                                                    "packageType", "packageVersion"):
                continue
            r = rows[norm]
            r["n"] += 1
            c = cn.get(k)
            if c is None:
                r["missing"] += 1
                if len(r["ex_missing"]) < 6:
                    r["ex_missing"].append(f"{repo}/{fn}::{k[:110]} EN={ev[:60]!r}")
                continue
            cv = c[1]
            if cv == ev:
                r["same"] += 1
                if len(r["ex_same"]) < 8:
                    r["ex_same"].append(f"{repo}/{fn}::{k[:110]} = {ev[:60]!r}")
            elif not CJK.search(cv) and LATIN.search(cv):
                r["nocjk"] += 1
                if len(r["ex_same"]) < 8:
                    r["ex_same"].append(f"NOCJK {repo}/{fn}::{k[:110]} EN={ev[:50]!r} CN={cv[:50]!r}")

out = {k: v for k, v in rows.items() if v["same"] or v["nocjk"] or v["missing"]}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "untranslated_by_class.json"),
                    "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(rows)} classes; {len(out)} with same/nocjk/missing\n")
for k, v in sorted(out.items(), key=lambda kv: -(kv[1]['same'] + kv[1]['nocjk'] + kv[1]['missing'])):
    print(f"{k}\n   n={v['n']} same={v['same']} nocjk={v['nocjk']} missing={v['missing']}")
    for e in v["ex_same"][:5]:
        print(f"      · {e}")
    for e in v["ex_missing"][:4]:
        print(f"      ? {e}")
