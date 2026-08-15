# -*- coding: utf-8 -*-
"""English-gate term lookup over the paired en/cn compendium leaf trees.

For each query term T (case-sensitive, word-ish boundary), walk the en tree and
the cn tree in lockstep by identical leaf path; whenever the EN leaf contains T,
report the CN leaf. Groups CN renderings by the Chinese substring that sits where
T sat is impossible in general, so we just print the (en, cn) leaf pairs, capped.

Anti-空转: prints, per term, how many en leaves were scanned and how many matched.
A term with scanned==0 means the corpus never loaded -> that is reported as
NO-CORPUS, not as "0 hits".
"""
import json, os, re, sys, collections

BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"
CN = os.path.join(BASE, "cn")
EN = os.path.join(BASE, "en")


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
    for fn in sorted(os.listdir(CN)):
        if not fn.endswith(".json"):
            continue
        enp = os.path.join(EN, fn)
        if not os.path.exists(enp):
            continue
        cn = json.load(open(os.path.join(CN, fn), encoding="utf-8"))
        en = json.load(open(enp, encoding="utf-8"))
        cnd = dict(leaves(cn))
        for path, s in leaves(en):
            pairs.append((fn, path, s, cnd.get(path)))
    return pairs


def main():
    terms = sys.argv[1:]
    pairs = load_pairs()
    print(f"corpus: {len(pairs)} en leaves paired across {len(set(p[0] for p in pairs))} files")
    if not pairs:
        print("NO-CORPUS")
        sys.exit(2)
    for t in terms:
        rx = re.compile(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])")
        hits = [(f, p, e, c) for (f, p, e, c) in pairs if rx.search(e)]
        withcn = [h for h in hits if h[3] and h[3] != h[2]]
        print(f"\n=== {t!r}  scanned={len(pairs)} en-hits={len(hits)} translated-pairs={len(withcn)}")
        seen = collections.Counter()
        for f, p, e, c in withcn:
            # short leaves are the informative ones (names/labels)
            if len(e) <= 90:
                seen[(e, c)] += 1
        for (e, c), n in seen.most_common(25):
            print(f"   [{n:>3}] EN: {e}\n         CN: {c}")
        if not seen:
            for f, p, e, c in withcn[:5]:
                idx = rx.search(e).start()
                print(f"   long-leaf sample {f}:{'/'.join(p)}\n     EN…{e[max(0,idx-60):idx+80]}…")
                if c:
                    print(f"     CN…{c[:200]}…")


if __name__ == "__main__":
    main()
