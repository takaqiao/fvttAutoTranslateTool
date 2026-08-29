#!/usr/bin/env python3
"""One unit from every residual item that still sits under `journals.`
（分类名、页名之类的零碎，`prep_units.py bucket` 按设计会把它们排除掉）。"""
import json, os, re, sys


def sole_entry(pack):
    """三个 Alien 包都是**单 Adventure 文档包**，顶层 entries 恰好一条，
    名字随包不同（Alien RPG System / Alien Evolved Starter Set /
    Alien Evolved Core Rules）。EC 版把 "Ember Early Access" 写死在调用处，
    换个包就静默返回空表。这里现取，取不到就当场喊。"""
    e = (pack or {}).get("entries") or {}
    if len(e) != 1:
        raise SystemExit(f"顶层 entries 有 {len(e)} 条，预期 1 条（单 Adventure 包）：{list(e)[:5]}")
    return next(iter(e.values()))


P = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
ROOT = os.environ.get("ALIEN_PARALLEL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parallel")
RES = os.path.join(P, "7-其他内容", "reports", "corerules", "todo", "_residual_after_fallback.json")
CNP = os.path.join(P, "3-核心书汉化插件", "compendium", "cn", "alien-evolved-corerules.alien-evolved-core-rules.json")
CJK = re.compile(r"[一-鿿]")

items = json.load(open(RES, encoding="utf-8"))["packs"]["alien-evolved-corerules.alien-evolved-core-rules"]
sel = [it for it in items if ".journals." in "." + it["path"]]
name = sys.argv[1] if len(sys.argv) > 1 else "journal-rest"
d = os.path.join(ROOT, name)
os.makedirs(d, exist_ok=True)
json.dump({"journal": "各卷零碎（分类名 / 页名 / 少量正文）", "items": sel},
          open(os.path.join(d, "todo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 锚点：每一卷已译好的分类名，让 agent 看得到同类字段的既有写法
cn = sole_entry(json.load(open(CNP, encoding="utf-8")))["journals"]
anchor = {}
for jn, j in cn.items():
    cats = j.get("categories") or {}
    done = {k: v for k, v in cats.items() if isinstance(v, str) and CJK.search(v)}
    if done:
        anchor[jn] = {"categories": done, "name": j.get("name")}
json.dump(anchor, open(os.path.join(d, "already_translated.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"{name}: {len(sel)} 条 / {sum(i['chars'] for i in sel)} 字符，锚点 {len(anchor)} 卷")
