# -*- coding: utf-8 -*-
"""
probe_channel_gap.py — 「补丁覆盖面不足」判据的**通道**维（只读）

前两个探针查的是「同一表面里翻了一半」；这个查「整条输出通道有没有被任何补丁碰过」。
方法：把 Ember 面向用户吐字的通道逐条列出，对每条问三件事
  1) 上游有多少条**字面英文**（非 i18n 键）；
  2) 同一通道上有多少条走了 i18n 键（说明这条通道「本可以译」，只是漏了那部分）；
  3) 汉化侧有没有任何机制能碰到它。

ui.notifications 是重点：Foundry v14 的 Notifications **不是 Application**
（client/applications/ui/notifications.mjs:30 `export default class Notifications`，
#render 直接 document.createElement("li") 塞进 #notifications），
所以 renderApplication/renderApplicationV2 钩子永远不会为它触发，
ember-hardcoded-cn.mjs 的 patchRenderedApplications 结构上够不着。
"""
import os
import re

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts"
FILES = ["ember.mjs", "crucible-async.mjs", "dnd5e-async.mjs"]

CALL = re.compile(r'ui\.notifications\.(info|warn|error|success|notify)\(\s*(.)', re.S)
LIT = re.compile(r'ui\.notifications\.(info|warn|error|success|notify)\(\s*([`"\'])(.*?)\2', re.S)

tot = {"literal": [], "i18nkey": [], "expr": []}
for f in FILES:
    s = open(os.path.join(EMBER, f), encoding="utf-8").read()
    lines = s.splitlines()
    for m in CALL.finditer(s):
        ln = s[:m.start()].count("\n") + 1
        head = s[m.start():m.start() + 260].replace("\n", " ")
        lm = LIT.match(s[m.start():])
        if not lm:
            tot["expr"].append((f, ln, head[:120]))
            continue
        text = lm.group(3)
        if re.match(r"^[A-Z][A-Z0-9_]*(\.[A-Za-z0-9_]+)+$", text.strip()):
            tot["i18nkey"].append((f, ln, text))
        else:
            tot["literal"].append((f, ln, text.replace("\n", " ")[:110]))

print("ui.notifications 调用统计")
for k in ("literal", "i18nkey", "expr"):
    print(f"  {k:<8} {len(tot[k])}")
print()
print("--- 走 i18n 键的（说明这条通道本来就能译）---")
for f, ln, t in tot["i18nkey"]:
    print(f"  {f}:{ln}  {t}")
print()
print("--- 字面英文（无论 lang 还是运行时补丁都碰不到）---")
for f, ln, t in tot["literal"]:
    print(f"  {f}:{ln}  {t}")
