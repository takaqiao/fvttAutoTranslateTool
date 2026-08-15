# -*- coding: utf-8 -*-
import os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f7_gate import rows, gate  # noqa

EXACT = [
    "Arcturian Alchemist", "Arcturian Jail", "Arcturian Liquor",
    "Arcturian Respirator", "Arcturian Sailor", "Arcturian Sea Captain",
    "Ordain Burial Grounds", "Ordain Docks", "Ordain Gazetteer", "Ordain Rumors",
    "Ordani Ruffian", "Pathways Gazetteer", "Pathways Scout Map",
    "Wandren Patroller", "Wandren Watcher", "Wandren HQ", "Not All Who Wandren",
    "Aedir Wellstone", "Silvered Aedir Warhammer", "Arcturel Investigation",
    "Altar of Orbis", "Well of Orbis", "Mayis Attunement", "The Gem of Mayis",
    "Temple of Ku'arta Key", "Kessian Embassy", "Kessian Sand Knife",
    "Tyraphic Asset", "Tyraphic Transformation", "Anydrath Training",
    "Boneway Stairs", "Hydroxol Hide", "Hydroxol Gills", "Gray Hydroxol Hide",
    "Signaran Opal", "Umber's Pass",
]
R = rows()
for s in EXACT:
    hits = {}
    for repo, fn, p, e, c in R:
        if e == s:
            hits.setdefault(c, []).append(p)
    if not hits:
        print("-- %-30r : ABSENT" % s)
        continue
    for c, ps in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        print("-- %-30r : %-38r x%-3d %s" % (s, c, len(ps),
              ps[0].replace("entries.Ember Early Access.", "")))

print()
for q in [
    {"label": "Arcturian", "en": "\\bArcturian\\b", "cn": ["阿克图里安", "阿克图里亚", "阿克图瑞安"]},
    {"label": "Ordain", "en": "\\bOrdain\\b", "cn": ["奥尔丹", "奥丹"]},
    {"label": "Ordani", "en": "\\bOrdani\\b", "cn": ["奥尔达尼", "奥达尼"]},
    {"label": "Pathways", "en": "\\bPathways\\b", "cn": ["通路", "通途", "通道区"]},
    {"label": "Wandren", "en": "\\bWandren\\b", "cn": ["万德伦", "旺德伦"]},
    {"label": "Aedir", "en": "\\bAedir\\b", "cn": ["艾迪尔", "艾狄尔", "埃迪尔"]},
]:
    gate(q, 0)
