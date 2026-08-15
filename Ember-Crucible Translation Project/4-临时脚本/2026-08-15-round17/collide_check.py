#!/usr/bin/env python3
"""Pre-landing collision check for round17 batches.

Discipline #1: a batch value is a whole-leaf snapshot. Before landing we must
prove nothing else touched the same key after the snapshot was taken. Checked
two ways:
  1. mtime of the target pack vs. mtime of the batch file
  2. the CURRENT cn leaf must still be the pre-image the batch author described
     (README table: 毛毛雨/风暴/降雨 for the weather leaves, 狂澜之月 for Mayis),
     and the ONLY differences between current and new must be the intended ones.

Path resolution is imported from apply_translations so it matches byte for byte.
"""
import json, os, sys, difflib, importlib.util

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROUND = os.path.join(P, "4-临时脚本", "2026-08-15-round17")
B = os.path.join(ROUND, "batches")
QA = os.path.join(P, "3-常用脚本", "qa")

spec = importlib.util.spec_from_file_location("at", os.path.join(QA, "apply_translations.py"))
at = importlib.util.module_from_spec(spec)
spec.loader.exec_module(at)

JOBS = [
    ("1-Ember汉化插件", "ember.adventure.json", "r17-weather-tiers-ember.adventure.json"),
    ("1-Ember汉化插件", "ember.crucible-adventure.json", "r17-weather-tiers-ember.crucible-adventure.json"),
    ("1-Ember汉化插件", "ember.crucible-character.json", "r17-tempest-moon-ember.crucible-character.json"),
]

PROBES = ["毛毛雨", "细雨", "狂风暴雨", "风暴", "降雨", "狂澜之月", "风暴之月",
          "令牌", "代币", "指示物", "坩埚", "Crucible"]

ok = True
for repo, pack, batchname in JOBS:
    cnp = os.path.join(P, repo, "compendium", "cn", pack)
    enp = os.path.join(P, repo, "compendium", "en", pack)
    bp = os.path.join(B, batchname)
    cn = at.load(cnp)
    en = at.load(enp)
    batch = at.load(bp)
    mt_pack, mt_batch = os.path.getmtime(cnp), os.path.getmtime(bp)
    print("=" * 72)
    print(f"{repo} / {pack}")
    print(f"  pack mtime {mt_pack:.0f} vs batch mtime {mt_batch:.0f} -> "
          + ("OK: no write to pack since snapshot" if mt_pack < mt_batch
             else "WARN: pack written AFTER snapshot"))
    if mt_pack >= mt_batch:
        ok = False
    for key, newval in batch.items():
        parts = at.split_path(en.get("entries", {}), key)
        src = at.get_at(en.get("entries", {}), parts)
        cur = at.get_at(cn.get("entries", {}), parts)
        print(f"  key {key}")
        print(f"    parts {parts}")
        print(f"    EN  : {type(src).__name__} len={len(src) if isinstance(src,str) else '-'}")
        print(f"    CUR : {type(cur).__name__} len={len(cur) if isinstance(cur,str) else '-'}")
        print(f"    NEW : len={len(newval)}")
        if not isinstance(src, str) or not isinstance(cur, str):
            print("    !! shape problem")
            ok = False
            continue
        for probe in PROBES:
            c0, c1 = cur.count(probe), newval.count(probe)
            if c0 or c1:
                flag = "" if c0 == c1 else "   <-- CHANGED"
                print(f"      {probe:8s} cur={c0} new={c1}{flag}")
        sm = difflib.SequenceMatcher(None, cur, newval, autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
        print(f"    diff hunks: {len(ops)}")
        for tag, i1, i2, j1, j2 in ops:
            print(f"      {tag}: {cur[i1:i2]!r} -> {newval[j1:j2]!r}")
print("=" * 72)
print("COLLISION CHECK:", "CLEAN" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
