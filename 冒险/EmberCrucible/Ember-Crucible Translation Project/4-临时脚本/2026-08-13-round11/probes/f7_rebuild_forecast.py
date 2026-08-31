# -*- coding: utf-8 -*-
"""Forecast what `3-常用脚本/tm/build_glossary.py` would write TODAY.

Replicates build_glossary's own harvest exactly -- same EN side
(5-其他内容/english-baseline/*), same walk_pairs, same is_term filter, same
majority vote -- so the ORPHAN verdicts are trustworthy.

ORPHAN = the key has no live (EN,CN) pair in the baselines, so it survives only
via the base glossary layer.  A rebuild can never repair those; they must be
hand-patched in BOTH glossary_ec.json and the base glossary.
"""
import json, os, re, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = "C:\\Users\\Taka\\Desktop\\fvtt\\Ember-Crucible Translation Project"
BASE_DIR = os.path.join(P, "5-\u5176\u4ed6\u5185\u5bb9", "english-baseline")
BASE_GLOSS = "C:\\Users\\Taka\\Desktop\\fvtt\\glossary_crucible_merged.json"
GLOSS = os.path.join(P, "5-\u5176\u4ed6\u5185\u5bb9", "glossary", "glossary_ec.json")

CJK = re.compile(r"[\u4e00-\u9fff]")
HTML = re.compile(r"<[^>]+>")
MAX_TERM_LEN = 60

HARVEST = [
    (os.path.join(BASE_DIR, "crucible-0.10.1"),
     os.path.join(P, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "compendium", "cn")),
    (os.path.join(BASE_DIR, "ember-0.6.0"),
     os.path.join(P, "1-Ember\u6c49\u5316\u63d2\u4ef6", "compendium", "cn")),
]


def is_term(s):
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) > MAX_TERM_LEN:
        return False
    if HTML.search(s) or "\n" in s:
        return False
    if s[-1] in ".!?:;,":
        return False
    return len(s.split()) <= 8


def walk_pairs(en, cn, out):
    if isinstance(en, dict) and isinstance(cn, dict):
        for k, v in en.items():
            if k in cn:
                walk_pairs(v, cn[k], out)
    elif isinstance(en, list) and isinstance(cn, list):
        for a, b in zip(en, cn):
            walk_pairs(a, b, out)
    elif isinstance(en, str) and isinstance(cn, str):
        out.append((en, cn))


pairs = defaultdict(lambda: defaultdict(int))
for en_dir, cn_dir in HARVEST:
    for fn in sorted(f for f in os.listdir(en_dir)
                     if f.endswith(".json") and not f.startswith("_")):
        cp = os.path.join(cn_dir, fn)
        if not os.path.isfile(cp):
            continue
        en_doc = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
        cn_doc = json.load(open(cp, encoding="utf-8"))
        got = []
        walk_pairs(en_doc.get("entries", {}), cn_doc.get("entries", {}), got)
        for e, c in got:
            if is_term(e) and CJK.search(c):
                pairs[e][c] += 1

harvested = {e: sorted(cs.items(), key=lambda kv: -kv[1])[0][0] for e, cs in pairs.items()}
g = json.load(open(GLOSS, encoding="utf-8"))
braw = json.load(open(BASE_GLOSS, encoding="utf-8"))
base = {k: v for k, v in braw.items() if isinstance(v, str) and CJK.search(v)}

KEYS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "f7_keys.json"), encoding="utf-8"))

print("%-32s %-24s %-24s %s" % ("key", "glossary now", "rebuild gives", "verdict"))
print("-" * 116)
orph, heal, agree = [], [], []
for k in KEYS:
    cur = g.get(k, "<<MISSING>>")
    if k in harvested:
        new = harvested[k]
        v = "already agrees" if new == cur else "REBUILD HEALS"
        (agree if new == cur else heal).append(k)
    else:
        new = "-- no live EN/CN pair --"
        v = "ORPHAN (base=%r)" % base.get(k, braw.get(k, "<absent>"))
        orph.append(k)
    print("%-32s %-24s %-24s %s" % (k[:32], cur[:24], new[:24], v))

print()
print("ORPHANS      %2d  %s" % (len(orph), orph))
print("REBUILD HEALS %2d" % len(heal))
print("already agrees %2d  %s" % (len(agree), agree))
