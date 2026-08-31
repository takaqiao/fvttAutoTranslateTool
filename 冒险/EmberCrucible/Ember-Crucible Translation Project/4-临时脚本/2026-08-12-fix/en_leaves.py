# -*- coding: utf-8 -*-
"""List every leaf whose ENGLISH equals one of the given strings, with pack+path+CN."""
import argparse, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SKIP_KEYS = {"_id", "path", "_variants", "_when"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--en", required=True, help="pipe-separated exact English strings")
    a = ap.parse_args()
    want = set(x for x in a.en.split("|") if x)
    rows = []
    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for p, e, c in sub:
                if e.strip() in want:
                    rows.append((fn, p, e, c))
    for e in sorted(want):
        print(f"=== {e!r}")
        for fn, p, en_, c in rows:
            if en_.strip() == e:
                print(f"   {fn:<32} {p}\n        -> {c}")


if __name__ == "__main__":
    main()
