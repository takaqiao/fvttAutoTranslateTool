# -*- coding: utf-8 -*-
"""Round-22 English gate for the 212 soundscape ARRANGEMENT labels.

Walks the paired en/cn compendium leaf trees of BOTH plugins (ember + crucible)
plus both lang files, and for each query term reports three tiers of evidence:
  * EXACT     : en leaf == term            -> the cn leaf (a name field: strongest)
  * SHORT     : en leaf short (<=60 chars) -> the cn leaf verbatim
  * BILINGUAL : anywhere in a cn leaf, the pattern `<中文…> [The ]<term>` — this
                project writes proper nouns bilingually in prose, so this pulls
                the established rendering straight out of running text.

Case-sensitive by design (this project has been burned four times by case-folding).

Anti-空转: prints, per run, how many en leaves were paired and how many files;
and per term how many leaves were scanned + how many matched. scanned==0 => exit 2.
Terms are read from a file (argv[1]), one per line, so no shell escaping is involved.
"""
import json, os, re, sys, collections, io

sys.stdout.reconfigure(encoding="utf-8")

ROOTS = [
    r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium",
    r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium",
]
LANGS = [
    (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\lang\en.json",
     r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\lang\cn.json"),
    (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\lang\en.json",
     r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\lang\cn.json"),
]

CJK = "\u4e00-\u9fff\u3400-\u4dbf"


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (str(i),))
    elif isinstance(o, str):
        yield path, o


def load_pairs():
    pairs = []
    for base in ROOTS:
        cnd_dir, end_dir = os.path.join(base, "cn"), os.path.join(base, "en")
        if not os.path.isdir(cnd_dir) or not os.path.isdir(end_dir):
            continue
        for fn in sorted(os.listdir(cnd_dir)):
            if not fn.endswith(".json"):
                continue
            enp = os.path.join(end_dir, fn)
            if not os.path.exists(enp):
                continue
            cn = json.load(io.open(os.path.join(cnd_dir, fn), encoding="utf-8"))
            en = json.load(io.open(enp, encoding="utf-8"))
            cnd = dict(leaves(cn))
            for path, s in leaves(en):
                pairs.append((fn, path, s, cnd.get(path)))
    for enp, cnp in LANGS:
        if not (os.path.exists(enp) and os.path.exists(cnp)):
            continue
        en = json.load(io.open(enp, encoding="utf-8"))
        cn = json.load(io.open(cnp, encoding="utf-8"))
        cnd = dict(leaves(cn))
        tag = "lang:" + os.path.basename(os.path.dirname(os.path.dirname(enp)))
        for path, s in leaves(en):
            pairs.append((tag, path, s, cnd.get(path)))
    return pairs


def main():
    terms = [l.rstrip("\n") for l in io.open(sys.argv[1], encoding="utf-8") if l.strip()]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pairs = load_pairs()
    files = sorted(set(p[0] for p in pairs))
    print(f"corpus: {len(pairs)} en leaves paired across {len(files)} files")
    if not pairs:
        print("NO-CORPUS")
        sys.exit(2)
    # every cn leaf, for the bilingual sweep
    cn_all = [c for (_f, _p, _e, c) in pairs if c]
    print(f"cn leaves available for bilingual sweep: {len(cn_all)}")
    for t in terms:
        rx = re.compile(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])")
        bil = re.compile(r"([" + CJK + r"·—’'A-Za-z0-9]{1,14})\s*(?:The\s+)?" + re.escape(t) + r"(?![A-Za-z])")
        exact, short, bilhits = collections.Counter(), collections.Counter(), collections.Counter()
        nhit = 0
        for f, p, e, c in pairs:
            if not rx.search(e):
                continue
            nhit += 1
            if c is None or c == e:
                continue
            if e.strip() == t:
                exact[c.strip()] += 1
            elif len(e) <= 60:
                short[f"{e.strip()}  ->  {c.strip()}"] += 1
        nb = 0
        for c in cn_all:
            for m in bil.finditer(c):
                head = m.group(1)
                if re.search("[" + CJK + "]", head):
                    bilhits[head] += 1
                    nb += 1
        print(f"\n=== {t!r}  scanned={len(pairs)} en-hits={nhit} exact={sum(exact.values())} "
              f"short={sum(short.values())} bilingual={nb}")
        for v, n in exact.most_common(cap):
            print(f"   EXACT     x{n}  {v}")
        for v, n in short.most_common(cap):
            print(f"   SHORT     x{n}  {v}")
        for v, n in bilhits.most_common(cap):
            print(f"   BILINGUAL x{n}  {v} | {t}")


main()
