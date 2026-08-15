# -*- coding: utf-8 -*-
import os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
pat = re.compile("接下来的一个回合|接下来一整轮|接下来的一轮|下一个回合")
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "node_modules")]
    for f in fn:
        if not f.lower().endswith((".json", ".js", ".mjs", ".md", ".txt")):
            continue
        p = os.path.join(dp, f)
        try:
            s = open(p, encoding="utf-8").read()
        except Exception:
            continue
        c = {}
        for m in pat.finditer(s):
            c[m.group(0)] = c.get(m.group(0), 0) + 1
        if c:
            print(f"{c}  {p}")
