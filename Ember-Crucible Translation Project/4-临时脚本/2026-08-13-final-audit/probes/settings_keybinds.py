# -*- coding: utf-8 -*-
"""
settings_keybinds.py —— 「注册进核心宿主 App 的模块设置 / 按键绑定」子判据。

宿主：SettingsConfig（Foundry client/applications/settings/config.mjs:11，id "settings-config"）
      与 ControlsConfig（client/applications/sidebar/apps/controls-config.mjs）。
两者根元素 class 与类名都不带 ember => ember-hardcoded-cn.mjs:453 的闸 return。
另一条通道：SettingsConfig 对 name/hint 做 _loc（config.mjs:126-127），
但 lang/cn.json 486 个键全带点号、无平铺键；而且 hint 串末尾带句点，
Localization#localize 走 getProperty 会按点号切路径，平铺键也救不了带句点的串。
"""
import re, os, json

EMB = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
LANG = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\lang\cn.json"
src = open(EMB, encoding="utf-8").read()
lang = json.load(open(LANG, encoding="utf-8"))

def balance(s, start, limit=4000):
    d = 0
    for i in range(start, min(len(s), start + limit)):
        if s[i] == "{": d += 1
        elif s[i] == "}":
            d -= 1
            if d == 0: return s[start:i+1]
    return s[start:start+limit]

rows = []
for kind, pat in [("setting", r'game\.settings\.register\(\s*"ember",\s*"(\w+)",\s*\{'),
                  ("keybinding", r'game\.keybindings\.register\(\s*"ember",\s*"(\w+)",\s*\{')]:
    for m in re.finditer(pat, src):
        blk = balance(src, m.end()-1)
        ln = src[:m.start()].count("\n") + 1
        cfg = re.search(r"config:\s*(true|false)", blk)
        nm = re.search(r'name:\s*"([^"]+)"', blk)
        hn = re.search(r'hint:\s*"([^"]+)"', blk)
        visible = (kind == "keybinding") or (cfg and cfg.group(1) == "true")
        if not (nm or hn): continue
        rows.append({"kind": kind, "key": m.group(1), "line": ln, "visible_in_ui": bool(visible),
                     "name": nm.group(1) if nm else None,
                     "hint": hn.group(1) if hn else None,
                     "name_in_lang": (nm.group(1) in lang) if nm else None,
                     "hint_in_lang": (hn.group(1) in lang) if hn else None})

vis = [r for r in rows if r["visible_in_ui"]]
n = sum(bool(r["name"]) + bool(r["hint"]) for r in vis)
print(f"带 name/hint 的 ember 注册项: {len(rows)}；UI 里可见的: {len(vis)}；可见英文串合计 {n} 条")
for r in vis:
    print(f"  [{r['kind']}] {r['key']} @{r['line']}")
    print(f"      name: {r['name']}  (lang 覆盖={r['name_in_lang']})")
    if r["hint"]: print(f"      hint: {r['hint']}  (lang 覆盖={r['hint_in_lang']})")
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings_keybinds.json")
json.dump(rows, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", dst)
