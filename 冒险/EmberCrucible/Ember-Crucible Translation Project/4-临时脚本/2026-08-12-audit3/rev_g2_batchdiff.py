# -*- coding: utf-8 -*-
"""G2 复核：批次逐条对当前 compendium/cn 做 diff，看哪些是真改动、哪些是幂等重放；
同时检查标记（@UUID / 内联命令 / class）与非 name 叶。"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
B = (r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt"
     r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3\batches")

jobs = [("G2__ember__ember.adventure.json", "1-Ember汉化插件", "ember.adventure.json"),
        ("G2__ember__ember.crucible-adventure.json", "1-Ember汉化插件", "ember.crucible-adventure.json"),
        ("G2__crucible__crucible.equipment.json", "2-Crucible汉化插件", "crucible.equipment.json")]

MARK = re.compile(r'@UUID\[[^\]]*\]|@Embed\[[^\]]*\]|\[\[[^\]]*\]\]|class="[^"]*"|&[A-Za-z]+\[')

for bf, repo, pack in jobs:
    b = json.load(open(os.path.join(B, bf), encoding="utf-8"))
    cn = json.load(open(os.path.join(P, repo, "compendium", "cn", pack), encoding="utf-8"))["entries"]
    en = json.load(open(os.path.join(P, repo, "compendium", "en", pack), encoding="utf-8"))["entries"]
    print(f"== {bf}  条数 {len(b)}")
    changed = same = 0
    for k, v in b.items():
        node, enode = cn, en
        for p in k.split("."):
            node = node.get(p) if isinstance(node, dict) else None
            enode = enode.get(p) if isinstance(enode, dict) else None
        if node == v:
            same += 1
        else:
            changed += 1
            print(f"   [改动] {k}")
            print(f"      EN  {enode!r}")
            print(f"      旧  {node!r}")
            print(f"      新  {v!r}")
        m = MARK.findall(str(v))
        if m:
            print(f"      标记 {m}")
    non_name = [k for k in b if not k.endswith(".name")]
    print(f"   幂等重放 {same} / 真改动 {changed}；非 name 叶 {non_name}")
    print(f"   最长值 {max(b.values(), key=len)!r}")
    print(f"   含 HTML/标记的值 {[v for v in b.values() if MARK.search(v) or '<' in v]}")
