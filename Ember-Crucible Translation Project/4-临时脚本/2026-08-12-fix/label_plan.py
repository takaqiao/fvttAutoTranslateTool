# -*- coding: utf-8 -*-
"""Produce a review worksheet for untranslated @UUID/[[..]] labels of one pack."""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CJK = re.compile(r'[一-鿿]')
SKIP_KEYS = {"_id", "path", "_variants", "_when"}
MARK = re.compile(r'(@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])(\{([^{}]*)\})?')


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--name-index", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=110)
    ap.add_argument("--grep-path")
    a = ap.parse_args()

    ni = json.load(open(a.name_index, encoding="utf-8"))
    lm = json.load(open(a.label_map, encoding="utf-8"))
    en = json.load(open(os.path.join(a.repo, "compendium", "en", a.pack), encoding="utf-8"))
    cn = json.load(open(os.path.join(a.repo, "compendium", "cn", a.pack), encoding="utf-8"))
    rows = []
    walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)

    out = []
    for p, e, c in rows:
        if c is None:
            continue
        if a.grep_path and not re.search(a.grep_path, p):
            continue
        cm = list(MARK.finditer(c))
        bad = [m for m in cm if m.group(3) and not CJK.search(m.group(3))]
        if not bad:
            continue
        em = list(MARK.finditer(e))
        # EN label attribution: prefer the EN markup with the SAME bracket target
        # (robust to reordered prose); fall back to positional.
        by_t = {}
        for m2 in em:
            by_t.setdefault(m2.group(1), []).append(m2.group(3))
        item = {"path": p, "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                "labels": [], "cn_full": c}
        for m in bad:
            i = cm.index(m)
            cand = by_t.get(m.group(1))
            if cand and len(set(x for x in cand if x)) == 1 and any(cand):
                enlab = next(x for x in cand if x)
            else:
                enlab = em[i].group(3) if i < len(em) else None
            s, t = max(0, m.start() - a.ctx), min(len(c), m.end() + a.ctx)
            d = {"idx": i, "target": m.group(1), "en": enlab, "cn": m.group(3),
                 "ctx": c[s:t]}
            if enlab in ni:
                d["name_idx"] = ni[enlab]
            if enlab in lm:
                d["label_map"] = lm[enlab]
            item["labels"].append(d)
        out.append(item)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"leaves={len(out)} labels={sum(len(x['labels']) for x in out)} -> {a.out}")


if __name__ == "__main__":
    main()
