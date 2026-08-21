#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 gaps.json 里的缺口按「宿主」分组：ember.mjs 里最近的顶层声明 / 模板文件。

前置自证：
  (A) 条数守恒 —— 分组后条数之和 = gaps.json 的键数（一条都不许丢）。
  (B) 归组正确 —— 已知的几条串必须落进已知的宿主。
"""
import json, re, os, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
U = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
MJS = os.path.join(U, "scripts", "ember.mjs")

src_lines = open(MJS, encoding="utf-8").read().split("\n")
RE_TOP = re.compile(r'^(?:export\s+)?(?:const|let|var|function|class|async function)\s+([A-Za-z_$][\w$]*)')
tops = []  # (lineno, name)
for i, ln in enumerate(src_lines, 1):
    m = RE_TOP.match(ln)
    if m: tops.append((i, m.group(1)))
top_lines = [t[0] for t in tops]
import bisect
def host_of(lineno):
    i = bisect.bisect_right(top_lines, lineno) - 1
    return tops[i][1] if i >= 0 else "<top>"

gaps = json.load(open(os.path.join(HERE, "gaps.json"), encoding="utf-8"))
groups = collections.defaultdict(list)
for s, locs in gaps.items():
    loc = locs[0]
    f, ln = loc.rsplit(":", 1)
    if f == "ember.mjs":
        groups[f"mjs::{host_of(int(ln))}"].append((s, loc))
    else:
        groups[f"tpl::{f}"].append((s, loc))

total = sum(len(v) for v in groups.values())
print(f"[自证A] 分组后 {total} 条 / gaps.json {len(gaps)} 条 -> " + ("OK" if total == len(gaps) else "FAIL"))
assert total == len(gaps)

CHECK = {"Hair Roots": "mjs::COLORS", "Elevator Destination": None}
print("[自证B] 抽样归组：")
for s in ["Hair Roots", "Hair Base", "Hair Sparkle", "Hair 1"]:
    where = [g for g, v in groups.items() if any(x[0] == s for x in v)]
    print(f"   {s!r} -> {where}")

print()
for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print(f"{len(v):5d}  {g}")

json.dump({g: [x[0] for x in v] for g, v in groups.items()},
          open(os.path.join(HERE, "gaps_by_host.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
