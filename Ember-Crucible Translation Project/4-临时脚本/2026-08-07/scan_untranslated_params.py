#!/usr/bin/env python3
"""Quoted parameter values inside markup that are still English.

`@Embed[Actor.x readaloud="…"]` renders that sentence to the player, but the
markup gate used to compare the whole marker verbatim, so translating it was
impossible -- translators照抄 English to get past the gate. The gate is fixed
now; this finds whatever the old rule left behind.
"""
from __future__ import annotations
import json
import os
import re

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CN_DIR = os.path.join(P, "1-Ember汉化插件", "compendium", "cn")
CJK = re.compile(r'[一-鿿]')
LATIN_WORD = re.compile(r'[A-Za-z][A-Za-z\'’\-]{2,}')
PARAM = re.compile(r'(\w+)=\s*"([^"]*)"')
MARKER = re.compile(r'@[A-Za-z]+\[[^\]]*\]')

rows = []


def walk(node, path, pack):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], pack)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], pack)
    elif isinstance(node, str):
        for m in MARKER.finditer(node):
            for name, val in PARAM.findall(m.group(0)):
                words = LATIN_WORD.findall(val)
                if len(words) >= 4 and not CJK.search(val):
                    rows.append((pack, '.'.join(path), name, len(val), val[:70]))


for fn in sorted(f for f in os.listdir(CN_DIR) if f.endswith('.json')):
    walk(json.load(open(os.path.join(CN_DIR, fn), encoding='utf-8')), [], fn)

print(f"{'pack':<32}{'参数':<12}{'字符':>6}  路径 / 开头")
for pack, path, name, n, head in rows:
    print(f"{pack[:31]:<32}{name:<12}{n:>6}  {path[:70]}")
    print(f"{'':<50}  {head}")
print(f"\n共 {len(rows)} 处未译的参数值，合计 {sum(r[3] for r in rows)} 字符")
