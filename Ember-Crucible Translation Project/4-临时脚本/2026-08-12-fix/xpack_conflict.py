# -*- coding: utf-8 -*-
"""For every leaf in <pack>, find leaves in <against> packs with the SAME English
but a DIFFERENT Chinese. English-gated by construction.

Only compares short leaves (names/labels) by default -- prose never matches byte
for byte across packs in a meaningful way.
"""
import argparse, json, os, sys, collections
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


def load(repo, fn):
    en = json.load(open(os.path.join(repo, "compendium", "en", fn), encoding="utf-8"))
    cp = os.path.join(repo, "compendium", "cn", fn)
    cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
    out = []
    walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--against", required=True, help="comma list of pack filenames")
    ap.add_argument("--max-en", type=int, default=60)
    a = ap.parse_args()
    mine = load(a.repo, a.pack)
    theirs = collections.defaultdict(collections.Counter)
    for fn in a.against.split(","):
        for p, e, c in load(a.repo, fn.strip()):
            if c:
                theirs[e.strip()][c] += 1
    n = 0
    for p, e, c in mine:
        if len(e) > a.max_en or not c:
            continue
        other = theirs.get(e.strip())
        if not other:
            continue
        if c in other:
            continue
        n += 1
        alts = ", ".join(f"{v!r}x{k}" for v, k in other.most_common(4))
        print(f"{p}\n   EN  {e!r}\n   MINE {c!r}\n   THEM {alts}")
    print(f"--- {n} conflicting leaves")


if __name__ == "__main__":
    main()
