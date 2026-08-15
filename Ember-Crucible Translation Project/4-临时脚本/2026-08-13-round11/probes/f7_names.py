# -*- coding: utf-8 -*-
"""Dump EN->CN for leaf paths matching a regex, across both repos."""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f7_gate import rows  # noqa

PATS = [
    r"folders\.(Earth|Fire|Water|Air|Mutagists|Toothbreakers|Signarans|Strayhearth)",
    r"pages\.Akonites\.name$",
    r"actors\.Mutagist [^.]+\.(name|tokenName)$",
    r"items\.Mutagist [^.]+\.name$",
    r"pages\.Mutagists\.name$",
    r"journals\.Toothbreaker Hideout\.name$",
    r"tables\.Toothbreaker [^.]+\.name$",
    r"actors\.Toothbreaker [^.]+\.(name|tokenName)$",
    r"items\.Toothbreaker [^.]+\.name$",
    r"items\.(Anachraenum Medallion|Lumarin Steel|Cheliceraeth Eye|Spectra's Blessing)\.name$",
    r"actors\.(Spectra|Kadhana Lizard|Young Cheliceraeth)\.(name|tokenName)$",
    r"pages\.Helkas Green\.name$",
    r"pages\.(The Arcageris|The Ordinate|Hallows|The Hallows)\.name$",
    r"pages\.(Signborn|Wirrun|Kivahr|Thornling|Cheliceraeth|Hulg'run|Vrjnhar)\.name$",
    r"items\.[^.]*Lineage\.name$",
    r"\.items\.(Signborn|Wirrun|Kivahr|Thornling|Hulg'run|Vrjnhar|Drakon)\.name$",
    r"tables\.Yakoshta Mine Track Switches\.name$",
    r"pages\.Cascilian\.(name|pronunciation)$",
]

seen = set()
R = rows()
for pat in PATS:
    rx = re.compile(pat)
    print("#### " + pat)
    out = {}
    for repo, fn, p, e, c in R:
        if rx.search(p):
            key = (p, e, c)
            if key in out:
                continue
            out[key] = fn
    for (p, e, c), fn in sorted(out.items()):
        print("   %s\n      EN %r\n      CN %r" % (p.replace("entries.Ember Early Access.", ""), e, c))
    print()
