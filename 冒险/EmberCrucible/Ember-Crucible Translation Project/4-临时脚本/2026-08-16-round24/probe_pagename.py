# -*- coding: utf-8 -*-
"""核实 `The Abyss` 到底是不是 emberCosmos00000 的 JournalEntryPage 名字。
用仓里已有的英文基准 / 中文 babele 映射作证据源（合集数据，不是 .mjs）。"""
import json, os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"

def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + "/[%d]" % i)
    else:
        yield path, o

TARGETS = ("The Abyss", "Heart of Ember", "Abyss")
for side in ("en", "cn"):
    for fn in os.listdir(os.path.join(P, side)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(P, side, fn), encoding="utf-8"))
        for path, v in walk(d):
            if not isinstance(v, str):
                continue
            if v.strip() in TARGETS or (side == "cn" and path.endswith("The Abyss")):
                print("[%s] %-42s %-70s = %r" % (side, fn, path[-70:], v))
