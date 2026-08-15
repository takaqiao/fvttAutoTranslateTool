# -*- coding: utf-8 -*-
"""
p13c_notify_split.py —— 把 ember 的硬编码通知按「是否 dnd5e 侧」切开。

ember.mjs 是 rollup 打包产物，dnd5e 整合与 crucible 整合的代码块交错，
行号区间不可靠。改用局部标识判定：命中点前后 60 行里出现
Actor5e / dnd5e / systems/dnd5e / "5e" 相关标识的，判为 dnd5e 侧（项目已定不管）。

假阳性模式：跨 120 行的上下文可能混入相邻块的 dnd5e 标识，导致把系统无关的条目
误判成 dnd5e 侧（偏保守，会低估 crucible 侧数量）。
只读。
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
t = open(P, encoding="utf-8", errors="replace").read()
L = t.split("\n")
RX = re.compile(r"ui\.notifications\.(info|warn|error|success)\(\s*[\"'`]([^\"'`]{4,200})")
D5 = re.compile(r"Actor5e|dnd5e|Item5e|RestConfiguration|dnd5e2")
KEY = re.compile(r"^[A-Z][A-Z0-9_]*\.[A-Za-z0-9_.]+$")

c5, cc, keys = [], [], []
for m in RX.finditer(t):
    ln = t.count("\n", 0, m.start()) + 1
    lit = m.group(2)
    if KEY.match(lit):
        keys.append((ln, lit))
        continue
    ctx = "\n".join(L[max(0, ln - 61):ln + 60])
    (c5 if D5.search(ctx) else cc).append((ln, m.group(1), lit))

print(f"lang-key 通知（已走 i18n，正常）: {len(keys)}")
print(f"硬编码英文 · dnd5e 侧（不管）: {len(c5)}")
print(f"硬编码英文 · crucible/系统无关侧: {len(cc)}")
print()
for ln, k, s in cc:
    print(f"{ln:7d} {k:5s} {s[:110]}")
print("\n--- dnd5e 侧 ---")
for ln, k, s in c5:
    print(f"{ln:7d} {k:5s} {s[:90]}")
