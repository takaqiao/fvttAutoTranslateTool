#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判据 P2 的**核实**面：对每个候选槽位，在上游语料里数「读取点」。

dead_write_slots.py 负责枚举（写在哪），本脚本负责判定（上游读不读）。
分开写是因为枚举可以纯正则，判定必须逐条给出**具体的上游读取点行号**，
否则「0 命中」既可能是真死也可能是 grep 写错了（FP2）。

每条 probe 都给：
  - reads: 上游确实读这个槽位的模式（命中数 > 0 ⇒ 写是活的）
  - decoys: 容易误当成读取点的模式（命中数只用于说明为什么不算）
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys

FOUNDRY = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
TREES = [
    (os.path.join(FOUNDRY, "client"), "core-js"),
    (os.path.join(FOUNDRY, "common"), "core-common"),
    (os.path.join(FOUNDRY, "templates"), "core-hbs"),
    (os.path.join(DATA, "systems", "crucible"), "crucible"),
    (os.path.join(DATA, "modules", "ember"), "ember"),
    (os.path.join(DATA, "modules", "babele"), "babele"),
]

PROBES = [
    # 槽位, 说明, 读取模式
    ("months.values[].name",
     "ember-hardcoded-cn.mjs:424 写 CONFIG.time.worldCalendarConfig / game.time.calendar 的月名",
     [r"months\.values\[[^\]]*\]\.name", r"month\.name", r"\bmonthName\b", r"\{\{[^}]*month[^}]*name"]),
    ("days.values[].name",
     "ember-hardcoded-cn.mjs:424 写星期名",
     [r"days\.values\[[^\]]*\]\.name", r"weekday\.name", r"\bdayName\b", r"\{\{[^}]*weekday"]),
    ("days.values[].abbreviation",
     "ember-hardcoded-cn.mjs:426 写星期缩写",
     [r"\.abbreviation\b"]),
    ("seasons.values[].name  (对照组：这才是日期串真正读的槽位)",
     "formatEmberDate 用它，且过 _loc()",
     [r"seasons\.values\[[^\]]*\]\.name", r"season\.name"]),
    ("months.values[].ordinal  (对照组：core 真正读的槽位)",
     "formatTimestamp 用 ordinal",
     [r"month\.ordinal", r"\.ordinal\b"]),
    ("ui.windows  写入者",
     "ember-hardcoded-cn.mjs:433 遍历 ui.windows 找日历应用重画",
     [r"ui\.windows\[", r"windows\[this\.appId\]"]),
    ("#ember-calendar 的 change 监听",
     "ember-hardcoded-cn.mjs:436 派发 new Event('change')",
     [r'addEventListener\("change"', r"addEventListener\('change'"]),
    ("i18n key `Sort`",
     "babele-register.js:57 写 game.i18n.translations.Sort",
     [r'localize\("Sort"\)', r"localize\('Sort'\)", r'localize\s+"Sort"', r'label:\s*"Sort"']),
    ("i18n key `sort`（小写）",
     "babele-register.js:58 写 game.i18n.translations.sort",
     [r'localize\("sort"\)', r"localize\('sort'\)", r'localize\s+"sort"', r'label:\s*"sort"']),
]


EXT = (".mjs", ".js", ".hbs", ".html")
_CACHE = {}


def _corpus(root):
    """(相对路径, 行号, 行文本) 全量缓存。不用 ripgrep —— 本机没有它，
    静默返回 0 命中会把「上游不读」的结论整个做假。"""
    if root in _CACHE:
        return _CACHE[root]
    out = []
    if os.path.isdir(root):
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".git", "node_modules", "packs", "assets", "icons", "fonts", "audio")]
            for f in fns:
                if not f.endswith(EXT):
                    continue
                p = os.path.join(dp, f)
                try:
                    for i, ln in enumerate(open(p, encoding="utf-8", errors="replace").read().splitlines(), 1):
                        out.append((os.path.relpath(p, root), i, ln))
                except Exception:
                    pass
    _CACHE[root] = out
    return out


def rg(pattern, root):
    rx = re.compile(pattern)
    return [f"{rel}:{i}: {ln.strip()}" for rel, i, ln in _corpus(root) if rx.search(ln)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    report = []
    for slot, why, pats in PROBES:
        total, sample = 0, []
        for root, tag in TREES:
            for pat in pats:
                hits = rg(pat, root)
                total += len(hits)
                sample += [f"[{tag}] {h[:150]}" for h in hits[:3]]
        verdict = "DEAD (上游 0 读取点)" if total == 0 else f"LIVE ({total} 读取点)"
        report.append(dict(slot=slot, why=why, hits=total, verdict=verdict, sample=sample[:8]))
        print(f"\n=== {slot}\n    {why}\n    -> {verdict}")
        for s in sample[:8]:
            print("      " + s)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dead_write_verify.json"),
         "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
