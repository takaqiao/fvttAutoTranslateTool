# -*- coding: utf-8 -*-
"""
notif_escape.py —— 「输出根本不落在任何 Application 根里」子判据。

v14 的 ui.notifications 不是 Application：
  client/applications/ui/notifications.mjs:30  `export default class Notifications {`
  构造函数只做 #initialize()，通知条由 notify() 直接 append 进 #notifications，
  **没有任何 render* 钩子会触发**。
所以 ember-hardcoded-cn.mjs 的 renderApplicationV2 / renderApplication 两个钩子
永远看不到通知文本；notify() 内部虽然调 _loc(message)（notifications.mjs:121），
但 lang/cn.json 的 486 个键全部是点分键、无平铺键，且多数通知串自带句点，
getProperty 会按点号切开路径，实际也查不中。=> 结构性无法覆盖。

本脚本枚举 ember.mjs 中所有 ui.notifications.* 的英文字面量，
并标出所属源码区段（scenes / dnd5e / crucible），供判断在 crucible 世界的可达性。
区段依据 rollup 拼接出的 `/** @module ember/xxx */` 标记行号。
"""
import re, os, json

EMB = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
src = open(EMB, encoding="utf-8").read()
lines = src.splitlines()

marks = [(m.start(), m.group(1)) for m in re.finditer(r"@module ember/(\w+)", src)]
mark_lines = [(src[:o].count("\n") + 1, n) for o, n in marks]

def region(ln):
    prev = "core"
    for l, n in mark_lines:
        if ln >= l: prev = n
        else: break
    return prev

rows = []
for m in re.finditer(r'ui\.notifications\.(warn|error|info|notify|success)\(\s*(?:"((?:[^"\\]|\\.)*)"|`((?:[^`\\]|\\.)*)`)',
                     src, re.S):
    s = (m.group(2) or m.group(3))
    ln = src[:m.start()].count("\n") + 1
    localize = bool(re.search(r'localize:\s*true', src[m.end():m.end()+160]))
    rows.append({"line": ln, "level": m.group(1), "region": region(ln),
                 "localize_opt": localize, "text": s.replace("\n", " ")})

from collections import Counter
print("ui.notifications 字面量总数:", len(rows))
print("按源码区段:", Counter(r["region"] for r in rows))
print("带 localize:true 的:", sum(1 for r in rows if r["localize_opt"]))
print("含 i18n 键形态的:", sum(1 for r in rows if re.match(r"^[A-Z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)+$", r["text"])))
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notif_escape.json")
json.dump(rows, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", dst)
