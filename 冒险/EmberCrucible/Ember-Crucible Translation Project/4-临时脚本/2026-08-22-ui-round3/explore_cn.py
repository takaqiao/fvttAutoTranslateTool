# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"
FIELDS = ("message", "description", "text", "revealedDialog", "unrevealedDialog", "effects")
for side in ("en", "cn"):
    for pack in ("ember.adventure", "ember.crucible-adventure"):
        d = json.load(open(rf"{BASE}\{side}\{pack}.json", encoding="utf-8"))
        rows = []
        nsc = nreg = nbeh = 0
        for ename, ev in d["entries"].items():
            sc = ev.get("scenes") or {}
            for sk, sv in sc.items():
                nsc += 1
                for rk, rv in (sv.get("regions") or {}).items():
                    nreg += 1
                    for bk, bv in (rv.get("behaviors") or {}).items():
                        nbeh += 1
                        for f in FIELDS:
                            if f in bv:
                                rows.append((sk, rk, bk, f, bv[f]))
        print(f"### {side}/{pack}: scenes={nsc} regions={nreg} behaviors={nbeh} hits={len(rows)}")
        for r in rows:
            print("   ", r[0], "|", r[1], "|", r[2], "|", r[3], "=", json.dumps(r[4], ensure_ascii=False)[:200])
