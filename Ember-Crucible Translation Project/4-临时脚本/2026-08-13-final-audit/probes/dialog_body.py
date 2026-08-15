# -*- coding: utf-8 -*-
"""
dialog_body.py —— 「DialogV2 正文/按钮在闸外」子判据（只读）

ember-hardcoded-cn.mjs:459-465 的 DialogV2 例外分支只改 `.window-title` 一个节点，
改完立刻 return。所以：
  * window.title    —— 若命中 EXACT 表就会被译；
  * content / ok.label / buttons[*].label —— **永远不会被翻**。
除非该 dialog 自己带了含 "ember" 的 class（那样 /ember/i.test(cls) 成立，
会走 translateNode 全树，但那时缺的是 EXACT 表词条，属另一类问题）。

本脚本抽取 ember.mjs 里两种 dialog 定义：
  (1) EmberInteractable 家族的 `static DEFAULT_CONFIG = ... dialog: {...}`
  (2) 直接调用 DialogV2.* 的内联 config
并对每个 dialog 报告：title / 是否带 ember class / content / 按钮标签 /
title 是否已在 CN 的 EXACT 表里。

假阳性模式：
  * 抽取靠花括号配平，遇到模板串里的 `{}` 可能截断，body 字段会显示原文供人工核对；
  * `label` 也可能是 i18n 键（如 "EMBER.X.Y"），脚本单列出来；
  * 部分 interactable 只在特定场景/开发模式出现，可达性需人工判断。
"""
import json, os, re

EMB = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CNJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"

src = open(EMB, encoding="utf-8").read()
cn = open(CNJ, encoding="utf-8").read()
EXACT = set(re.findall(r'^\s*"([^"]+)":\s*"[^"]*",?\s*$', cn, re.M))
I18N = re.compile(r"^[A-Z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)+$")

def balance(s, start):
    """从 s[start] 的 '{' 起取到配平的 '}'。"""
    d = 0
    for i in range(start, min(len(s), start + 4000)):
        if s[i] == "{": d += 1
        elif s[i] == "}":
            d -= 1
            if d == 0: return s[start:i+1]
    return s[start:start+1500]

rows = []
for m in re.finditer(r"\bdialog:\s*\{", src):
    blk = balance(src, m.end() - 1)
    ln = src[:m.start()].count("\n") + 1
    title = re.search(r'title:\s*"([^"]+)"', blk)
    content = re.search(r'content:\s*(?:"([^"]*)"|`([^`]*)`)', blk)
    labels = re.findall(r'label:\s*"([^"]+)"', blk)
    has_ember_class = bool(re.search(r'classes:\s*\[[^\]]*ember', blk, re.I))
    rows.append({
        "line": ln,
        "title": title.group(1) if title else None,
        "title_in_EXACT": (title.group(1) in EXACT) if title else None,
        "content": (content.group(1) or content.group(2)) if content else None,
        "button_labels": labels,
        "labels_are_i18n": [bool(I18N.match(l)) for l in labels],
        "has_ember_class": has_ember_class
    })

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialog_body.json")
json.dump(rows, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

n_bodyeng = sum(1 for r in rows if (r["content"] or r["button_labels"]) and not r["has_ember_class"])
n_title_cn = sum(1 for r in rows if r["title_in_EXACT"])
print(f"interactable dialog 定义: {len(rows)}")
print(f"  标题已被 EXACT 覆盖(即会显示中文标题): {n_title_cn}")
print(f"  正文/按钮有英文且无 ember class(闸外): {n_bodyeng}")
print(f"  英文按钮标签总数: {sum(len(r['button_labels']) for r in rows if not r['has_ember_class'])}")
print("->", dst)
