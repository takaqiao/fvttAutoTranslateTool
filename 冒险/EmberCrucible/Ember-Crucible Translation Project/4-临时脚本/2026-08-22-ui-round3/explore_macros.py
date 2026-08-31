# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"
for side in ("en","cn"):
    for pack in ("ember.adventure","ember.crucible-adventure"):
        d=json.load(open(rf"{BASE}\{side}\{pack}.json",encoding="utf-8"))
        for ename,ev in d["entries"].items():
            ms=ev.get("macros") or {}
            print(f"### {side}/{pack} macros={len(ms)}")
            for mk,mv in ms.items():
                print("   ", repr(mk), "keys=", list(mv.keys()), "name=", json.dumps(mv.get("name"),ensure_ascii=False))
