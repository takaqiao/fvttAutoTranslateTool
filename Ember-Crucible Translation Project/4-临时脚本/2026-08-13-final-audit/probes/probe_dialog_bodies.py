# -*- coding: utf-8 -*-
"""
probe_dialog_bodies.py —— M2 支线：DialogV2 的 **正文与按钮** 是否落在闸的可达域外。

判据（ember-hardcoded-cn.mjs:453-465）：
  根元素 class 不含 "ember" 且类名不以 "Ember" 开头时，
  只有 `root.querySelector(".window-title")` 这一个节点会被 translateText，
  **content / buttons / form 一律不进 translateNode**。
  → 因此这些位置的英文即便加进 EXACT 表也永远不生效（结构性不可达）。

本脚本把每个 DialogV2 调用的完整上下文（含 `const content = ...` 这类先赋值再传入的写法）
打出来，便于逐条人工核实。v1 的正则版会漏掉变量写法，这一版补上。
只读上游源码。
"""
import re
import sys
from pathlib import Path

SRC = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs")
T = SRC.read_text(encoding="utf-8", errors="replace")
L = T.splitlines()

DLG = re.compile(r"DialogV2\s*\.\s*(confirm|prompt|wait|query)\s*\(")
BEFORE = int(sys.argv[1]) if len(sys.argv) > 1 else 14
AFTER = int(sys.argv[2]) if len(sys.argv) > 2 else 26

for m in DLG.finditer(T):
    ln = T.count("\n", 0, m.start()) + 1
    lo, hi = max(0, ln - 1 - BEFORE), min(len(L), ln - 1 + AFTER)
    print("=" * 78)
    print(f"### ember.mjs:{ln}  {m.group(1)}")
    for i in range(lo, hi):
        print(f"{i+1:>7} {L[i]}")
