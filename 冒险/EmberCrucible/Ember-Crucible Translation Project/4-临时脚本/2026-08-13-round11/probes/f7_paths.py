# -*- coding: utf-8 -*-
"""List every pack leaf still carrying a given Chinese rendering (paths only)."""
import os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f7_gate import rows  # noqa

TARGETS = ["阿纳克雷努姆", "阿纳克拉埃努姆", "突变派", "螯蛛以太兽", "鲁玛林",
           "丝珀特拉", "斯佩克特拉", "赫尔卡斯·格林", "塞夫赫尔", "轨道道岔"]

R = rows()
for t in TARGETS:
    hits = [(fn, p) for repo, fn, p, e, c in R if t in (c or "")]
    print("== %s  (%d leaves)" % (t, len(hits)))
    for fn, p in hits:
        print("   %-30s %s" % (fn.replace(".json", ""), p.replace("entries.Ember Early Access.", "")))
    print()

print("== Spectra goddess-context leaves ==")
rx = re.compile(r"[Gg]oddess.{0,60}Spectra|Spectra.{0,60}[Gg]oddess", re.S)
for repo, fn, p, e, c in R:
    if rx.search(e):
        m = rx.search(e)
        print("   %s::%s" % (fn.replace(".json", ""), p.replace("entries.Ember Early Access.", "")))
        print("      EN %s" % e[max(0, m.start() - 20):m.end() + 20].replace("\n", " "))
        print("      CN? 光谱=%s 丝珀特拉=%s 斯佩克特拉=%s" % ("光谱" in c, "丝珀特拉" in c, "斯佩克特拉" in c))
