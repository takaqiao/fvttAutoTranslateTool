# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"
for pack in ["ember.adventure.json", "ember.crucible-adventure.json"]:
    for side in ["en", "cn"]:
        d = json.load(open(os.path.join(ROOT, side, pack), encoding="utf-8"))
        acts = d["entries"]["Ember Early Access"]["actors"]
        for n in ["Kali Andrella", "Agrimage", "Eveis Brightstone"]:
            it = acts[n]["items"]["Control Water"]
            print("="*90)
            print(pack, side, n, "keys=", list(it.keys()) if isinstance(it, dict) else type(it))
            if isinstance(it, dict):
                for k, v in it.items():
                    if isinstance(v, str):
                        s = v
                        m = re.search(r".{140}(next|following) round.{80}", s, re.I|re.S)
                        print(f"  [{k}] len={len(s)} hasNextRound={bool(re.search(r'(next|following) round', s, re.I))}")
                        if m: print("    ...", m.group(0).replace("\n"," "))
                        idx = s.find("Part Water")
                        if idx < 0: idx = s.find("分水")
                        if idx >= 0: print("    PARTWATER:", s[idx:idx+420].replace("\n"," "))
                    else:
                        print(f"  [{k}] type={type(v).__name__}")
