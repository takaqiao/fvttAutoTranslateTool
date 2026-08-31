# -*- coding: utf-8 -*-
import os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f7_gate import rows  # noqa

EXACT = [
    "House Cevher", "House Cevher Mausoleum", "House Cevher Signet Ring",
    "Funar Cevher", "Lyla Cevher", "A Cevher Summons",
    "Cascillian Autotool", "Cascillian Marine Officer", "Cascillian Rebreather",
    "Cascillian Republic", "Mutagist", "Mutagist Scout", "Mutagist Excisor",
    "Mutagist Grenadier", "Mutagist Clothing", "Mutagist Vivisector",
    "Mutagist Bombardier", "Toothbreaker Hideout", "Signborn", "Thornling",
    "Wirrun", "Kivahr", "Hulg'run", "Vrjnhar", "Signborn Lineage",
    "Wirrun Lineage", "Kivahr Lineage", "Thornling Lineage",
    "Hulg'run Lineage", "Vrjnhar Lineage", "Mutagen", "Mutagenic Affliction",
    "Vial of Mutagenic Medium", "Mutagenic Formulae",
]
R = rows()
for s in EXACT:
    hits = {}
    for repo, fn, p, e, c in R:
        if e == s:
            hits.setdefault(c, []).append(p)
    if not hits:
        print("-- %-32r : ABSENT from packs" % s)
        continue
    print("-- %-32r :" % s, end="")
    first = True
    for c, ps in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        pad = "" if first else " " * 39
        print("%s %-40r x%-3d  %s" % (pad, c, len(ps),
              ps[0].replace("entries.Ember Early Access.", "")))
        first = False
