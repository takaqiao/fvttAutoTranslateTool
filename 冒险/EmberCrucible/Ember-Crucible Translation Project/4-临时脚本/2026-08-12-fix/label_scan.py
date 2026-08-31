# -*- coding: utf-8 -*-
"""Scan @UUID[...]{label} / [[...]]{label} labels: find untranslated (no-CJK) ones,
and build an EN-label -> CN-label majority map from labels that ARE translated.

Pairing is positional within a leaf (i-th markup in EN vs i-th in CN); the packs
already pass the markup-multiset gate so counts match. A handful of leaves have
shuffled labels (audit 1.4) -- those become minority noise in the map and are
filtered by the support threshold.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CJK = re.compile(r'[一-鿿㐀-䶿]')
SKIP_KEYS = {"_id", "path", "_variants", "_when"}

# @Foo[target]{label}  or  [[/cmd ...]]{label}
MARK = re.compile(r'(@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])(\{([^{}]*)\})?')


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))


def leaves(repo):
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
        cp = os.path.join(cn_dir, fn)
        cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
        out = []
        walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], out)
        for p, e, c in out:
            yield fn, p, e, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--out-map")
    ap.add_argument("--out-todo")
    a = ap.parse_args()

    pairs = []          # (repo, pack, path, idx, target, en_label, cn_label)
    for repo in a.repo:
        rn = os.path.basename(repo.rstrip("/\\"))
        for fn, p, e, c in leaves(repo):
            if c is None:
                continue
            em = MARK.findall(e)
            cm = MARK.findall(c)
            if len(em) != len(cm):
                continue
            for i, ((et, _, el), (ct, _, cl)) in enumerate(zip(em, cm)):
                if not el and not cl:
                    continue
                pairs.append((rn, fn, p, i, ct, el, cl))

    # majority map from translated labels
    votes = defaultdict(Counter)
    for rn, fn, p, i, ct, el, cl in pairs:
        if el and cl and CJK.search(cl):
            votes[el][cl] += 1

    mp = {}
    for el, c in votes.items():
        top, n = c.most_common(1)[0]
        mp[el] = {"cn": top, "n": n, "total": sum(c.values()),
                  "alts": {k: v for k, v in c.items() if k != top}}

    todo = [dict(repo=rn, pack=fn, path=p, idx=i, target=ct, en=el, cn=cl)
            for rn, fn, p, i, ct, el, cl in pairs
            if cl and not CJK.search(cl)]

    have = sum(1 for t in todo if t["en"] in mp)
    print(f"markup-with-label pairs : {len([1 for x in pairs if x[5]])}")
    print(f"untranslated CN labels  : {len(todo)}   (map-covered {have}, new {len(todo)-have})")
    byp = Counter((t['repo'], t['pack']) for t in todo)
    for k, v in byp.most_common():
        print(f"   {k[0]:<22} {k[1]:<34} {v}")

    if a.out_map:
        json.dump(mp, open(a.out_map, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.out_todo:
        json.dump(todo, open(a.out_todo, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
