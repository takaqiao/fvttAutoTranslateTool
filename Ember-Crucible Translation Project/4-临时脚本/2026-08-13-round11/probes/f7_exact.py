# -*- coding: utf-8 -*-
"""Look up exact EN strings (whole-leaf equality) and short-context greps."""
import os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f7_gate import rows  # noqa

EXACT = [
    "Mutagist Bombadier", "Mutagist Contingent", "Mutagist Scouts", "Mutagists",
    "Toothbreakers", "Toothbreaker Rumors", "Signaran Opal", "Signarans",
    "Earth", "Earth Elemental", "Earth Potency", "Earth Spellcraft", "Rune: Earth",
    "Akonites", "Akon", "Umber's Pass", "Cruel Dragons", "Anachraenum Member",
    "Helkas Green", "Kadhana Lizard", "The Arcageris", "The Ordinate",
    "Lumarin Steel", "Strayhearth Caravan", "Eternas", "Cascilian", "Arcden",
    "Elder Goddess Spectra", "Shrine to Spectra", "Tomb of Spectra's Chosen",
    "Spectra", "Luma", "Draconic", "Highgate", "Marlstone",
    "Afflicted Thornling", "Strange Thornlings", "Young Cheliceraeth",
    "Cheliceraeth Eye", "The Signborn's Secret", "Cascal Arcden", "Moiran",
]

R = rows()
print("===== EXACT whole-leaf EN matches =====")
for s in EXACT:
    hits = {}
    for repo, fn, p, e, c in R:
        if e == s:
            hits.setdefault(c, []).append(p)
    print("-- %r : %d distinct CN" % (s, len(hits)))
    for c, ps in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        short = ps[0].replace("entries.Ember Early Access.", "")
        print("     %-46r x%-4d e.g. %s" % (c, len(ps), short))

print()
print("===== Languages table row for Luma / Draconic =====")
rx = re.compile(r"pages\.Languages\.text$")
for repo, fn, p, e, c in R:
    if rx.search(p) and "ember.adventure" in fn:
        for m in re.finditer(r"(Luma|Draconic|Arcden|Cascal|Eonic)", e):
            a = max(0, m.start() - 90)
            print("   EN ...%s..." % e[a:m.end() + 40].replace("\n", " "))
        for m in re.finditer(r"(卢玛|龙语|奥克登|阿克登|卡斯卡尔|伊欧尼克|永世)", c):
            a = max(0, m.start() - 45)
            print("   CN ...%s..." % c[a:m.end() + 30].replace("\n", " "))
        break
