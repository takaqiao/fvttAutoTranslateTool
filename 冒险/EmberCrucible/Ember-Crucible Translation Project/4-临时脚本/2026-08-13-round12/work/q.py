# -*- coding: utf-8 -*-
import json, re, sys, io, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROWS = json.load(open(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round12\work\ALL.json", encoding="utf-8"))

ap = argparse.ArgumentParser()
ap.add_argument("--en"); ap.add_argument("--cn"); ap.add_argument("--path")
ap.add_argument("--count-cn", help="comma list of CN substrings to count")
ap.add_argument("--show", type=int, default=0)
ap.add_argument("--trunc", type=int, default=200)
a = ap.parse_args()

rs = ROWS
if a.en: rx = re.compile(a.en); rs = [r for r in rs if rx.search(r["en"])]
if a.cn: rx = re.compile(a.cn); rs = [r for r in rs if r["cn"] and rx.search(r["cn"])]
if a.path: rx = re.compile(a.path); rs = [r for r in rs if rx.search(r["path"])]
print(f"matched leaves: {len(rs)}")
if a.count_cn:
    for w in a.count_cn.split(","):
        n = sum(r["cn"].count(w) for r in rs if r["cn"])
        L = sum(1 for r in rs if r["cn"] and w in r["cn"])
        print(f"  CN {w!r}: {n} occurrences in {L} leaves")
for r in rs[: a.show]:
    print("-" * 80)
    print(f"[{r['repo']}] {r['pack']} :: {r['path']}")
    print("E| " + r["en"][: a.trunc])
    print("C| " + (r["cn"] or "")[: a.trunc])
