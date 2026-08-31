# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
files = [
 r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round9\findings\same_en_split.json",
 r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round9\findings\REAL_DEFECTS.json",
 r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-final-audit\findings\xp_p5_round_turn.json",
]
for f in files:
    if not os.path.isfile(f):
        print("MISSING", f); continue
    s = open(f, encoding="utf-8").read()
    print("#"*90); print(f, "len", len(s))
    print("  'Control Water' count:", s.count("Control Water"),
          " 'Kali Andrella':", s.count("Kali Andrella"),
          " '接下来一整轮':", s.count("接下来一整轮"),
          " '接下来的一个回合':", s.count("接下来的一个回合"))
    for m in re.finditer("接下来一整轮|接下来的一个回合", s):
        print("   ctx:", s[max(0,m.start()-500):m.start()+120].replace("\n"," ")[-620:])
