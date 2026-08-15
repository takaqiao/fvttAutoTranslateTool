# -*- coding: utf-8 -*-
"""
probe: i18n_button_slot_ember —— ember 侧「按钮 / 窗口标题 / 对话框」这三个**必过 i18n** 的槽（只读）

ember 的 scripts/ember.mjs 是 13 万行打包产物，`label:` 在瓦片、预制件、颜色表里满天飞，
所以泛用的 `label:` 判据会吐 1800 条噪声。本探针把范围收到三个**结构上确定**的位置：

    buttons.push({… label: "…"})      → templates/generic/form-footer.hbs:10 `{{localize button.label}}`
    buttons: [{… label: "…"}]         → 同上
    ok/yes/no/confirm: {label: "…"}   → DialogV2 按钮，client/applications/api/dialog.mjs `_loc(label)`
    window: {title: "…"}              → ApplicationV2#title `get title(){return _loc(...)}`

覆盖侧要减四张表：ember/lang/en.json、本项目 lang/cn.json、core en.json、
以及本项目 scripts/ember-hardcoded-cn.mjs 的运行时替换表（它是按渲染后的 DOM 文本匹配的）。

假阳性模式：
  - `window: {…}` 也可能是别的意思的对象字面量（本探针要求同对象里出现 title/label）；
  - 运行时替换表的匹配发生在 DOM 上、且有「只处理 ember 自己的界面 + DialogV2 标题」这道闸，
    所以「表里有这条」不等于「这条一定被译到」，反过来「表里没有」是可靠的未覆盖信号。

只读。
"""
import io
import json
import os
import re
import sys

FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_button_slot_ember.json")


def flat(o, p=""):
    out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            q = (p + "." + k) if p else k
            out.add(q)
            out |= flat(v, q)
    return out


def main():
    src = os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs")
    s = io.open(src, encoding="utf-8").read()
    EN = flat(json.load(io.open(os.path.join(FVTT, "modules", "ember", "lang", "en.json"), encoding="utf-8")))
    CN = flat(json.load(io.open(os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json"), encoding="utf-8")))
    COREK = flat(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))
    hc = io.open(os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"), encoding="utf-8").read()
    rt = set(re.findall(r'"([^"\\]{2,120})"\s*:\s*"', hc))
    known = EN | CN | COREK | rt

    keyish = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$-]+)+$")
    lab = re.compile(r'\b(label|title|tooltip)\s*:\s*"((?:[^"\\]|\\.)*)"')
    pats = [
        ("buttons.push", re.compile(r"buttons\.push\(\s*\{([^{}]*)\}")),
        ("buttons:[", re.compile(r"buttons\s*:\s*\[([^\[\]]*)\]")),
        ("dialog-btn", re.compile(r"\b(?:ok|yes|no|reject|confirm)\s*:\s*\{([^{}]*)\}")),
        ("window", re.compile(r"window\s*:\s*\{([^{}]*)\}")),
    ]
    out = {}
    for name, rx in pats:
        for m in rx.finditer(s):
            for lm in lab.finditer(m.group(1)):
                t = lm.group(2).strip()
                if not t or t in known or keyish.match(t) or len(t) < 3:
                    continue
                ln = s[:m.start()].count("\n") + 1
                out.setdefault(f"{lm.group(1)}|{t}", []).append([name, ln])
    for k, v in sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"{k[:64]:<66} x{len(v):<3} {v[0][0]} @{v[0][1]}")
    print("distinct", len(out))
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
