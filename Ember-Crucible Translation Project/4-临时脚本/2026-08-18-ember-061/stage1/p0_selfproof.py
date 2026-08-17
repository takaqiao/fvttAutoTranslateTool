# -*- coding: utf-8 -*-
"""P0: 前置自证。
两件都要断言：
  (a) 切出来的条数 = 已知真值（gone 314 / changed 334 / added 2061）
  (b) 切对地方：gone 路径确实在 EN060 有、EN061 无；changed 的 old==EN060、new==EN061；
      added 路径确实 EN060 无、EN061 有。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

TRUTH = {
    "ember.adventure.json":          {"added": 804,  "gone": 40,  "changed": 129},
    "ember.crucible-adventure.json": {"added": 1255, "gone": 274, "changed": 204},
    "ember.crucible-adversary.json": {"added": 2,    "gone": 0,   "changed": 1},
}

delta = load_delta()
ok = True

# (a) counts
print("=== (a) 条数自证 ===")
tot = {"added": 0, "gone": 0, "changed": 0}
for f, buckets in TRUTH.items():
    for b, n in buckets.items():
        got = len(delta[f][b])
        tot[b] += got
        flag = "OK " if got == n else "MISMATCH"
        if got != n:
            ok = False
        print(f"  {flag} {f:32s} {b:8s} 切出 {got:5d} / 真值 {n:5d}")
print(f"  合计 added={tot['added']} gone={tot['gone']} changed={tot['changed']}")
assert tot == {"added": 2061, "gone": 314, "changed": 334}, tot
print("  合计与真值 2061/314/334 相等  OK")

# (b) 切对地方
print("\n=== (b) 切对地方自证（逐条回查 EN060 / EN061 真身） ===")
for f in FILES:
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))
    d = delta[f]
    bad = {"gone_not_in_060": 0, "gone_still_in_061": 0,
           "chg_old_ne_060": 0, "chg_new_ne_061": 0,
           "add_in_060": 0, "add_not_in_061": 0}
    for ptr in d["gone"]:
        parts = split_path(ptr)
        if not has_leaf(en060, parts):
            bad["gone_not_in_060"] += 1
        if has_leaf(en061, parts):
            bad["gone_still_in_061"] += 1
    for ptr, pair in d["changed"].items():
        parts = split_path(ptr)
        old, new = pair
        if get_at(en060, parts) != old:
            bad["chg_old_ne_060"] += 1
        if get_at(en061, parts) != new:
            bad["chg_new_ne_061"] += 1
    for ptr in d["added"]:
        parts = split_path(ptr)
        if has_leaf(en060, parts):
            bad["add_in_060"] += 1
        if not has_leaf(en061, parts):
            bad["add_not_in_061"] += 1
    st = "OK " if all(v == 0 for v in bad.values()) else "BAD"
    if st == "BAD":
        ok = False
    print(f"  {st} {f:32s} {bad}")

# (c) gone/changed 在 CN 侧的命中情况（不是断言，是盘点）
print("\n=== (c) CN 侧命中盘点 ===")
for f in FILES:
    cn = jload(os.path.join(CN, f))
    d = delta[f]
    g_hit = sum(1 for p in d["gone"] if has_at(cn, split_path(p)))
    c_hit = sum(1 for p in d["changed"] if has_at(cn, split_path(p)))
    a_hit = sum(1 for p in d["added"] if has_at(cn, split_path(p)))
    print(f"  {f:32s} gone 在CN {g_hit}/{len(d['gone'])} | changed 在CN {c_hit}/{len(d['changed'])} | added 已在CN {a_hit}/{len(d['added'])}")

print("\nSELFPROOF", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
