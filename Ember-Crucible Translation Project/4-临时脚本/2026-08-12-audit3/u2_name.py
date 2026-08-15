# -*- coding: utf-8 -*-
"""Print every `name`/`label` leaf whose ENGLISH value equals a given string, with its CN value + path."""
import argparse, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
SKIP = {"_id", "path", "_variants", "_when"}

def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP: continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True, help="exact English value")
    ap.add_argument("--any-field", action="store_true", help="not only name/label leaves")
    a = ap.parse_args()
    for repo in REPOS:
        ed = os.path.join(ROOT, repo, "compendium", "en")
        cd = os.path.join(ROOT, repo, "compendium", "cn")
        for fn in sorted(os.listdir(ed)):
            if not fn.endswith(".json") or fn.startswith("_"): continue
            en = json.load(open(os.path.join(ed, fn), encoding="utf-8"))
            cp = os.path.join(cd, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            rows = []
            walk(en, cn, [], rows)
            for p, e, c in rows:
                if e != a.en: continue
                last = p.split(".")[-1]
                if not a.any_field and last not in ("name", "label", "tokenName"): continue
                print(f"{repo[:1]} {fn} :: {p}\n     EN={e!r}\n     CN={c!r}")
main()
