# -*- coding: utf-8 -*-
"""
dialog_strings.py —— 汇总「会渲染进 DialogV2、但两条汉化通道都够不着」的英文串。

通道回顾（v14 实测源码依据）：
  1) window.title  -> ApplicationV2 get title() = game.i18n.localize(options.window.title)
     （client/applications/api/application.mjs:319-321）；localize 走 getProperty，
     串里没有 "." 时等价于查平铺键。=> **可以**被 lang 平铺键覆盖。
  2) buttons[].label / ok.label -> DialogV2._renderButtons: span.innerText = _loc(label)
     （client/applications/api/dialog.mjs:249）。=> 同样**可以**被 lang 平铺键覆盖。
  3) content -> _initializeApplicationOptions 只做 cleanHTML，**不 localize**
     （dialog.mjs:186-192）。=> 只能靠 DOM 替换。
而 ember-hardcoded-cn.mjs:459-465 的 DialogV2 分支只改 .window-title 后立刻 return，
所以 (2)(3) 都不进入 DOM 替换；lang/cn.json 里 486 个键**全部带点号、0 个平铺键**，
所以 (1)(2) 也没被 lang 覆盖。结论：三类里只有已进 EXACT 表的 title 会显示中文。

抽取范围：
  a) `dialog: { ... }` 平衡块（EmberInteractable 家族 DEFAULT_CONFIG）
  b) `_configureDialog` / `_displayDialog` 方法体
  c) `DialogV2.{prompt,confirm,wait,input}` 调用实参
排除：串本身是 i18n 键（含点号且首字母大写的点分路径）、FA 图标类、路径。
"""
import json, os, re

EMB = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CNJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
LANG = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\lang\cn.json"

src = open(EMB, encoding="utf-8").read()
cn = open(CNJ, encoding="utf-8").read()
lang = json.load(open(LANG, encoding="utf-8"))
EXACT = set(re.findall(r'^\s*"([^"]+)":\s*"[^"]*",?\s*$', cn, re.M))
I18N = re.compile(r"^[A-Z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)+$")

def balance(s, start, limit=6000):
    d = 0
    for i in range(start, min(len(s), start + limit)):
        if s[i] == "{": d += 1
        elif s[i] == "}":
            d -= 1
            if d == 0: return s[start:i+1], i
    return s[start:start+limit], start+limit

blocks = []
for m in re.finditer(r"\bdialog:\s*\{", src):
    b, _ = balance(src, m.end()-1)
    blocks.append(("DEFAULT_CONFIG.dialog", src[:m.start()].count("\n")+1, b))
for m in re.finditer(r"(?:async\s+)?_(?:configureDialog|displayDialog)\s*\([^)]*\)\s*\{", src):
    b, _ = balance(src, m.end()-1)
    blocks.append(("_configureDialog/_displayDialog", src[:m.start()].count("\n")+1, b))
for m in re.finditer(r"DialogV2[$\w]*\.(?:prompt|confirm|wait|input)\s*\(\s*\{", src):
    b, _ = balance(src, m.end()-1)
    blocks.append(("DialogV2call", src[:m.start()].count("\n")+1, b))

labels, contents, titles = {}, {}, {}
for kind, ln, b in blocks:
    ember_class = bool(re.search(r'classes:\s*\[[^\]]*ember', b, re.I))
    for mm in re.finditer(r'\blabel:\s*"([^"]+)"', b):
        s = mm.group(1)
        if I18N.match(s): continue
        labels.setdefault(s, []).append((kind, ln, ember_class))
    for mm in re.finditer(r'\bcontent:\s*(?:"([^"]*)"|`([^`]*)`)', b):
        s = (mm.group(1) or mm.group(2)).strip()
        if not s or not re.search(r"[A-Za-z]{3}", s): continue
        contents.setdefault(s, []).append((kind, ln, ember_class))
    for mm in re.finditer(r'\btitle:\s*(?:"([^"]+)"|`([^`]+)`)', b):
        s = (mm.group(1) or mm.group(2)).strip()
        if I18N.match(s) or s.startswith("_loc"): continue
        titles.setdefault(s, []).append((kind, ln, ember_class))

def report(name, d, localizable):
    gated = {k: v for k, v in d.items() if not any(e for _, _, e in v)}
    print(f"\n== {name}: {len(d)} 个不同串，其中 {len(gated)} 个所在 dialog 无 ember class（闸外）")
    for k in sorted(gated):
        covered = (k in EXACT) or (localizable and k in lang)
        flag = "[已覆盖]" if covered else "         "
        print(f"  {flag} {k!r}  @ {sorted({ln for _, ln, _ in gated[k]})}")
    return gated

g1 = report("按钮/ok 标签 label（可被 lang 平铺键救，但 lang 无平铺键）", labels, True)
g2 = report("dialog content（lang 完全够不着，只能 DOM）", contents, False)
g3 = report("window.title（EXACT 表已覆盖一部分）", titles, True)

out = {"labels": {k: [list(x) for x in v] for k, v in g1.items()},
       "contents": {k: [list(x) for x in v] for k, v in g2.items()},
       "titles": {k: [list(x) for x in v] for k, v in g3.items()}}
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialog_strings.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n->", dst)
