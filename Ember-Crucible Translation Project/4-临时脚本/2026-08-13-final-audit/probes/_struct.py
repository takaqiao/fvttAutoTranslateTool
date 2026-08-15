# -*- coding: utf-8 -*-
"""Read-only structure probe."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
p = os.path.join(ROOT, r"1-Ember汉化插件\compendium\en\ember.crucible-adventure.json")
d = json.load(open(p, encoding="utf-8"))
a = d["entries"]["Ember Early Access"]
for k, v in a.items():
    if isinstance(v, dict):
        print(f"{k}: dict[{len(v)}]", list(v.keys())[:8])
    elif isinstance(v, list):
        print(f"{k}: list[{len(v)}]")
    else:
        print(f"{k}: {type(v).__name__} {str(v)[:80]}")
