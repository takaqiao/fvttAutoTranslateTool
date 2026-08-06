#!/usr/bin/env python3
"""Orphan CN *pages* inside journals that still exist (upstream renamed the page,
not the journal).

Matching by name is useless here -- the name is exactly what changed. So match by
structure instead: a translation and its source share paragraph/list/section
counts and, more tellingly, the same multiset of Foundry markup (UUIDs and inline
commands are copied verbatim into the translation, so they survive as a
fingerprint).

Prints, for every orphan page, the untranslated EN page of the same journal whose
markup fingerprint it matches best.
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
PACK = "ember.crucible-adventure.json"
CJK = re.compile(r'[一-鿿]')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]')
TAG = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)')

en = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", PACK), encoding="utf-8"))
cn = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", PACK), encoding="utf-8"))
EJ = en["entries"]["Ember Early Access"]["journals"]
CJ = cn["entries"]["Ember Early Access"]["journals"]


def fingerprint(s: str) -> Counter:
    return (Counter(MARKUP.findall(s))
            + Counter(f'<{a}{b.lower()}' for a, b in TAG.findall(s)))


def similarity(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


rows = []
for jn, cj in CJ.items():
    ej = EJ.get(jn)
    if not ej:
        continue                                   # whole journal renamed: other script
    en_pages = ej.get("pages") or {}
    cn_pages = cj.get("pages") or {}
    # EN pages of this journal that nobody has translated yet
    open_pages = {pn: p.get("text", "") for pn, p in en_pages.items()
                  if not CJK.search((cn_pages.get(pn) or {}).get("text") or "")}
    for pn, p in cn_pages.items():
        if pn in en_pages:
            continue
        text = p.get("text") or ""
        if not CJK.search(text):
            continue
        fp = fingerprint(text)
        best, score = None, 0.0
        for opn, otext in open_pages.items():
            s = similarity(fp, fingerprint(otext))
            if s > score:
                best, score = opn, s
        rows.append((jn, pn, len(text), best, score,
                     len(open_pages)))

rows.sort(key=lambda r: -r[2])
print(f"{'journal':<28}{'orphan CN page':<34}{'chars':>7}  best untranslated EN page (markup overlap)")
for jn, pn, n, best, score, nopen in rows:
    flag = "  <== 强匹配" if score >= 0.7 else ("  ~ 待人工判断" if score >= 0.4 else "")
    print(f"{jn[:27]:<28}{pn[:33]:<34}{n:>7}  {str(best)[:38]:<40}{score:.2f}{flag}")
print(f"\n{len(rows)} 个孤儿页面")
