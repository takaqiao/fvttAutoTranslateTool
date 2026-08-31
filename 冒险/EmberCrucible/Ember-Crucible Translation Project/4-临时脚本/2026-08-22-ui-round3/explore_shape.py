# -*- coding: utf-8 -*-
"""Exploratory: learn the shape of scenes/regions/behaviors in the en baseline."""
import json, sys, io
sys.stdout.reconfigure(encoding='utf-8')
BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"
for pack in ("ember.adventure", "ember.crucible-adventure"):
    d = json.load(open(rf"{BASE}\en\{pack}.json", encoding="utf-8"))
    ents = d["entries"]
    print("=== pack", pack, "entries:", list(ents.keys()))
    for ename, ev in ents.items():
        sc = ev.get("scenes")
        print("  entry", ename, "scenes type", type(sc).__name__, "n", len(sc) if sc else 0)
        if isinstance(sc, dict):
            k0 = list(sc.keys())[:2]
            for kk in k0:
                print("    scene key:", repr(kk), "fields:", list(sc[kk].keys()))
        nreg = 0; nbeh = 0; behkeys = {}
        it = sc.items() if isinstance(sc, dict) else enumerate(sc or [])
        for sk, sv in it:
            regs = sv.get("regions")
            if not regs: continue
            rit = regs.items() if isinstance(regs, dict) else enumerate(regs)
            for rk, rv in rit:
                nreg += 1
                bs = rv.get("behaviors")
                if not bs: continue
                bit = bs.items() if isinstance(bs, dict) else enumerate(bs)
                for bk, bv in bit:
                    nbeh += 1
                    for f in bv.keys():
                        behkeys[f] = behkeys.get(f, 0) + 1
        print("    regions", nreg, "behaviors", nbeh, "behavior field counts", behkeys)
