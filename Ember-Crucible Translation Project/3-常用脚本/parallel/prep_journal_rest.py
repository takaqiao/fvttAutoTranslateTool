#!/usr/bin/env python3
"""One unit from every residual item that still sits under `journals.`
（分类名、页名之类的零碎，`prep_units.py bucket` 按设计会把它们排除掉）。"""
import json, os, re, sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROOT = os.environ.get("EMBER_PARALLEL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parallel")
RES = os.path.join(P, "5-其他内容", "reports", "ember", "todo", "_residual_after_fallback.json")
CNP = os.path.join(P, "1-Ember汉化插件", "compendium", "cn", "ember.crucible-adventure.json")
CJK = re.compile(r"[一-鿿]")

items = json.load(open(RES, encoding="utf-8"))["packs"]["ember.crucible-adventure"]
sel = [it for it in items if ".journals." in "." + it["path"]]
name = sys.argv[1] if len(sys.argv) > 1 else "journal-rest"
d = os.path.join(ROOT, name)
os.makedirs(d, exist_ok=True)
json.dump({"journal": "各卷零碎（分类名 / 页名 / 少量正文）", "items": sel},
          open(os.path.join(d, "todo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 锚点：每一卷已译好的分类名，让 agent 看得到同类字段的既有写法
cn = json.load(open(CNP, encoding="utf-8"))["entries"]["Ember Early Access"]["journals"]
anchor = {}
for jn, j in cn.items():
    cats = j.get("categories") or {}
    done = {k: v for k, v in cats.items() if isinstance(v, str) and CJK.search(v)}
    if done:
        anchor[jn] = {"categories": done, "name": j.get("name")}
json.dump(anchor, open(os.path.join(d, "already_translated.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"{name}: {len(sel)} 条 / {sum(i['chars'] for i in sel)} 字符，锚点 {len(anchor)} 卷")
