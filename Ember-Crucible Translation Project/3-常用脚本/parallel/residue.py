#!/usr/bin/env python3
"""Residual English in a batch file: Latin words left standing after tags,
Foundry markup and {labels} are stripped.

  python residue.py <batch.json>
"""
import json, re, sys

TAG = re.compile(r"<[^>]+>")
MARKUP = re.compile(r"@[A-Za-z]+\[[^\]]*\](\{[^}]*\})?|\[\[[^\]]*\]\](\{[^}]*\})?|&amp;[Rr]eference\[[^\]]*\](\{[^}]*\})?")
WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]{1,}")
# 全库既有写法，不算残留
ALLOW = {"AC", "DC", "HP", "GM", "PC", "XP", "gp", "sp", "cp", "pp", "GP", "SP",
         "he", "him", "she", "her", "they", "them", "his", "hers", "theirs",
         "CG", "CN", "CE", "LG", "LN", "LE", "NG", "NE", "gt", "lt", "amp", "nbsp",
         "For", "Other", "Fortunes", "With", "Our", "Own", "Two", "Hands", "Alt", "lb"}

batch = json.load(open(sys.argv[1], encoding="utf-8-sig"))
items = batch.get("items", batch)
total = 0
for path, v in items.items():
    # 条目名/页名/分类名按规范就是「中文 English」双语并列，不是残留
    if path.endswith(".name") or ".categories." in path:
        continue
    plain = MARKUP.sub(" ", TAG.sub(" ", v)).replace("&amp;", " ")
    words = [w for w in WORD.findall(plain) if w not in ALLOW]
    if words:
        total += len(words)
        print(f"{path}\n    {words[:20]}")
print(f"\nresidual latin words: {total}")
