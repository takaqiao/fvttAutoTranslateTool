# -*- coding: utf-8 -*-
"""U1 helper: for one @UUID target (or an EN name regex), census every CN label
used library-wide plus every CN `name`/`label` field of the matching documents.

  python u1_survey.py --target Item.w7jXkPu7MheM6bkw
  python u1_survey.py --en-name "Rune-Marked"
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SKIP_KEYS = {"_id", "path", "_variants", "_when"}
CJK = re.compile(r'[一-鿿]')
MARK = re.compile(r'@UUID\[([^\]]*)\](?:\{([^{}]*)\})?')


def leaves():
    """yield (repo, pack, dotted_path, en_str, cn_str_or_None)"""
    for repo in ("1-Ember汉化插件", "2-Crucible汉化插件"):
        en_dir = os.path.join(P, repo, "compendium", "en")
        cn_dir = os.path.join(P, repo, "compendium", "cn")
        if not os.path.isdir(en_dir):
            continue
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            stack = [(en.get("entries", {}), cn.get("entries", {}), [])]
            while stack:
                e, c, path = stack.pop()
                if isinstance(e, dict):
                    for k, v in e.items():
                        if k in SKIP_KEYS:
                            continue
                        stack.append((v, c.get(k) if isinstance(c, dict) else None, path + [str(k)]))
                elif isinstance(e, list):
                    for i, v in enumerate(e):
                        stack.append((v, c[i] if isinstance(c, list) and i < len(c) else None, path + [str(i)]))
                elif isinstance(e, str):
                    yield repo, fn, ".".join(path), e, (c if isinstance(c, str) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--en-name", default=None, help="regex matched against EN name/label fields")
    ap.add_argument("--en-label", default=None, help="regex matched against EN @UUID labels")
    a = ap.parse_args()

    lab = defaultdict(Counter)          # target -> Counter(cn label)
    lab_where = defaultdict(list)
    names = defaultdict(Counter)        # en name -> Counter(cn name)
    name_where = defaultdict(list)
    enlab = defaultdict(Counter)        # en label (matched) -> Counter(cn label)

    tre = set(a.target)
    nre = re.compile(a.en_name) if a.en_name else None
    lre = re.compile(a.en_label) if a.en_label else None

    for repo, pack, path, e, c in leaves():
        seg = path.split(".")[-1]
        if nre and seg in ("name", "label") and nre.search(e) and c:
            names[e][c] += 1
            name_where[e].append(f"{pack}:{path}")
        if not c:
            continue
        if "@UUID[" not in e:
            continue
        eb = list(MARK.finditer(e))
        cb = {}
        for m in MARK.finditer(c):
            cb.setdefault(m.group(1), []).append(m.group(2))
        for m in eb:
            t, el = m.group(1), m.group(2)
            cls = cb.get(t) or []
            for cl in cls:
                if t in tre:
                    lab[t][cl] += 1
                    lab_where[t].append(f"{pack}:{path}  EN={{{el}}}")
                if lre and el and lre.search(el):
                    enlab[el][cl] += 1

    for t in a.target:
        print(f"### target {t}")
        for k, v in lab[t].most_common():
            print(f"    {v:4d}  {k}")
        for w in lab_where[t][:40]:
            print(f"        {w}")
    if nre:
        print("### CN name fields matching EN name regex")
        for e, c in sorted(names.items()):
            print(f"    EN {e!r}")
            for k, v in c.most_common():
                print(f"        {v:4d}  {k}")
            for w in name_where[e][:4]:
                print(f"          @ {w}")
    if lre:
        print("### CN labels by EN label")
        for e, c in sorted(enlab.items()):
            print(f"    EN {{{e}}}  -> " + " | ".join(f"{k}×{v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
