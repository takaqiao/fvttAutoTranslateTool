# -*- coding: utf-8 -*-
"""Check whether Control Water / Kali Andrella exist in the twin pack ember.adventure.json"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"
for pack in ["ember.adventure.json", "ember.crucible-adventure.json"]:
    for side in ["en", "cn"]:
        p = os.path.join(ROOT, side, pack)
        d = json.load(open(p, encoding="utf-8"))
        ents = d.get("entries", {})
        for ename, ev in ents.items():
            acts = ev.get("actors", {}) if isinstance(ev, dict) else {}
            names = [a for a in acts if a in ("Kali Andrella", "Agrimage", "Eveis Brightstone", "Slaith")]
            print(f"{pack} [{side}] entry={ename} actors_total={len(acts)} target_actors={names}")
            for n in names:
                items = acts[n].get("items", {}) if isinstance(acts[n], dict) else {}
                cw = [k for k in items if "Control Water" in k or "控制水" in k]
                print(f"    {n}: items={len(items)} controlwater_keys={cw}")
