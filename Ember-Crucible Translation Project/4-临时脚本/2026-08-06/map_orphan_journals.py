#!/usr/bin/env python3
"""Where did the CN-only journals go?

`Arcturel Lower` / `Arcturel Upper` exist in compendium/cn but not in the 0.6.0
English baseline: upstream reorganised them. Their pages hold v1.0.15-era
Chinese that Babele can no longer apply. This matches each orphan page, by page
NAME, against every page of every current journal, and reports how the legacy
Chinese compares in size to today's English -- the deciding factor for whether
porting it is a head start or a trap (stage 13: the already-translated 65% is
missing whole blocks relative to current English).
"""
from __future__ import annotations
import json
import os
import re

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPO = os.path.join(P, "1-Ember汉化插件")
PACK = "ember.crucible-adventure.json"
TAG = re.compile(r'<[^>]+>')
CJK = re.compile(r'[一-鿿]')

en = json.load(open(os.path.join(REPO, "compendium", "en", PACK), encoding="utf-8"))
cn = json.load(open(os.path.join(REPO, "compendium", "cn", PACK), encoding="utf-8"))
EN = en["entries"]["Ember Early Access"]["journals"]
CN = cn["entries"]["Ember Early Access"]["journals"]


def textlen(s):
    return len(TAG.sub(' ', s)) if isinstance(s, str) else 0


# page name -> [(journal, en_text_len, cn_translated?)]
index = {}
for jn, j in EN.items():
    for pn, p in (j.get("pages") or {}).items():
        index.setdefault(pn, []).append((jn, textlen(p.get("text", "")), p))

for orphan in [j for j in CN if j not in EN]:
    pages = (CN[orphan].get("pages") or {})
    print(f"\n########## {orphan}  ({len(pages)} pages) ##########")
    hit = miss = 0
    for pn, p in pages.items():
        cn_len = textlen(p.get("text", ""))
        cands = index.get(pn, [])
        if not cands:
            miss += 1
            print(f"  [no page of this name today] {pn}  (CN {cn_len})")
            continue
        hit += 1
        for jn, en_len, en_page in cands:
            # is that live page already translated?
            live = (CN.get(jn, {}).get("pages") or {}).get(pn, {}).get("text")
            state = "已译" if isinstance(live, str) and CJK.search(live) else "未译"
            ratio = (cn_len / en_len) if en_len else 0
            print(f"  {pn[:34]:<36} -> {jn[:26]:<28} EN {en_len:>6}  legacyCN {cn_len:>6} "
                  f"({ratio:.0%})  live:{state}")
    print(f"  -- matched {hit}, unmatched {miss}")
