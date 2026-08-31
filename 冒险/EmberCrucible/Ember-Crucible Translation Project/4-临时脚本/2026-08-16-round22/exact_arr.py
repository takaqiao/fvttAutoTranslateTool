# -*- coding: utf-8 -*-
"""Round-22: does the compendium already contain an EN leaf that is *literally*
one of the 212 arrangement labels? (Many of them are scene / journal-page names.)
If so the cn leaf at the identical path is the established rendering — reuse it
verbatim rather than inventing one.

Anti-空转: prints corpus size + how many of the 212 were probed; a corpus of 0
or a label list of 0 => exit 2.
"""
import json, os, sys, io, collections

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "g", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_arr.py"))

# reuse the loader without running gate_arr's main(): re-declare it here instead
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
print(f"corpus: {len(pairs)} en leaves")
if not pairs:
    sys.exit(2)

by_en = collections.defaultdict(collections.Counter)
for f, p, e, c in pairs:
    if c and c != e:
        by_en[e.strip()][c.strip()] += 1

labels = json.load(io.open(os.path.join(BASE, r"4-临时脚本\2026-08-16-round22\soundscapes_r22.json"),
                           encoding="utf-8"))["arrLabels"]
print(f"probing {len(labels)} arrangement labels against {len(by_en)} distinct translated en leaves")
if not labels:
    sys.exit(2)
hit = 0
for lab in labels:
    if lab in by_en:
        hit += 1
        print(f"HIT  {lab}  ->  " + " | ".join(f"{v}(x{n})" for v, n in by_en[lab].most_common(4)))
print(f"\nliteral hits: {hit}/{len(labels)}")

# also: labels minus a trailing structural word
STRUCT = ["Day", "Night", "Tension", "Calm", "Main", "Intense", "Relaxed", "Interlude",
          "Rises", "Interval", "Heroic", "Atonal", "Spooky", "Weird", "Dramatic",
          "Verse", "Chorus", "Bridge", "Melody", "Rhythm", "Sad", "Quiet", "Chaos"]
print("\n--- stem hits (label minus one trailing structural word) ---")
h2 = 0
for lab in labels:
    for s in STRUCT:
        if lab.endswith(" " + s):
            stem = lab[: -(len(s) + 1)].rstrip(" -")
            if stem in by_en:
                h2 += 1
                print(f"STEM {lab}  [{stem}]  ->  " + " | ".join(f"{v}(x{n})" for v, n in by_en[stem].most_common(3)))
            break
print(f"stem hits: {h2}")
