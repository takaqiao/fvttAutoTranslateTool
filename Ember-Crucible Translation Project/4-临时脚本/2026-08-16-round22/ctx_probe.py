# -*- coding: utf-8 -*-
"""Round-22: raw context dump. For each term, print EVERY en leaf that contains it
together with the cn leaf, windowed around the match on both sides. Unlike
gate_arr.py this does not filter by leaf length, so it also surfaces terms whose
only occurrences are inside long prose.

Anti-空转: prints corpus size; exits 2 if 0. Per term prints the hit count even
when it is 0, so "no output" can never be mistaken for "not scanned".
"""
import json, os, re, sys, io

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROOTS = [os.path.join(BASE, r"1-Ember汉化插件\compendium"),
         os.path.join(BASE, r"2-Crucible汉化插件\compendium")]


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (str(i),))
    elif isinstance(o, str):
        yield path, o


pairs = []
for base in ROOTS:
    cnd_dir, end_dir = os.path.join(base, "cn"), os.path.join(base, "en")
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
print(f"corpus: {len(pairs)} en leaves", flush=True)
if not pairs:
    sys.exit(2)

W = 120
for t in [l.rstrip("\n") for l in io.open(sys.argv[1], encoding="utf-8") if l.strip()]:
    rx = re.compile(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])")
    hits = [(f, p, e, c) for (f, p, e, c) in pairs if rx.search(e)]
    print(f"\n########## {t!r}  hits={len(hits)}", flush=True)
    seen = set()
    for f, p, e, c in hits:
        m = rx.search(e)
        enw = e[max(0, m.start() - W):m.end() + W]
        if enw in seen:
            continue
        seen.add(enw)
        print(f"  --[{f}] {'/'.join(p[-3:])}")
        print(f"    EN: ...{enw}...")
        print(f"    CN: {c if c is not None else '(no cn leaf)'}"[:600])
