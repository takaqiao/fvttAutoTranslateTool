# -*- coding: utf-8 -*-
"""
probe: ember_interactable_dialogs —— Ember 场景可交互物件（EmberInteractable / EmberSwitch）
                                     的对话框标题与按钮标签（只读）

为什么单列：
    ember.mjs:62766 `_displayDialog(config)` → `DialogV2.wait(config)` / `DialogV2.prompt(config)`；
    ember.mjs:62782 `_configureDialog(config)` 把 `dialog.buttons` 这个 **对象**
    `{0:{label,icon},1:{…}}` 摊成 DialogV2 的按钮数组，并给 `config.window.title` 兜底。
    core `client/applications/api/dialog.mjs:249` 对每个按钮做 `span.innerText = _loc(label)`；
    `ApplicationV2#title` 对 `window.title` 做 `_loc(...)`。
    ⇒ 这些 label / title **都在 i18n 通道上**，只是上游写的是裸英文、没注册键。

    这一族是**按 DEFAULT_CONFIG 声明**的，一个物件一条，散落在 13 万行里，
    既有的 `label:` 泛用判据会被瓦片资源数据淹掉，所以要按 `dialog: {…}` 块做花括号配平提取。

覆盖侧要减：ember en.json / 本项目 cn.json / core en.json / 本项目运行时替换表。
注意本项目的运行时补丁（ember-hardcoded-cn.mjs:445 patchRenderedApplications）对 DialogV2
**只改 .window-title**，按钮文字与正文根本不进那条路径 —— 所以即使标题进了表，按钮也仍是英文。

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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ember_interactable_dialogs.json")


def flat(o, p=""):
    out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            q = (p + "." + k) if p else k
            out.add(q)
            out |= flat(v, q)
    return out


def block(s, i):
    """从 s[i] == '{' 起做花括号配平（忽略字符串内的括号），返回块结束下标。"""
    depth = 0
    j = i
    n = len(s)
    while j < n:
        c = s[j]
        if c in "\"'`":
            q = c
            j += 1
            while j < n and s[j] != q:
                if s[j] == "\\":
                    j += 1
                j += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return n


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

    titles, labels = {}, {}
    lit = re.compile(r'\b(title|label)\s*:\s*"((?:[^"\\]|\\.)*)"')
    for m in re.finditer(r"\bdialog\s*:\s*\{", s):
        i = m.end() - 1
        j = block(s, i)
        blk = s[i:j + 1]
        ln0 = s[:i].count("\n") + 1
        for lm in lit.finditer(blk):
            kind, t = lm.group(1), lm.group(2).strip()
            if not t or keyish.match(t):
                continue
            bucket = titles if kind == "title" else labels
            bucket.setdefault(t, []).append(ln0 + blk[:lm.start()].count("\n"))

    # _configureDialog 里动态塞的 label / title
    dyn = {}
    for m in re.finditer(r"_configureDialog\s*\([^)]*\)\s*\{", s):
        i = m.end() - 1
        j = block(s, i)
        blk = s[i:j + 1]
        ln0 = s[:i].count("\n") + 1
        for lm in lit.finditer(blk):
            t = lm.group(2).strip()
            if not t or keyish.match(t):
                continue
            dyn.setdefault(f"{lm.group(1)}|{t}", []).append(ln0 + blk[:lm.start()].count("\n"))

    def dump(name, d):
        cov = {k: v for k, v in d.items() if k in known}
        gap = {k: v for k, v in d.items() if k not in known}
        print("=" * 92)
        print(f"{name}: 共 {len(d)} 个不同串；已被四张覆盖表兜住 {len(cov)}；**未覆盖 {len(gap)}**")
        for k, v in sorted(gap.items()):
            print(f"   [GAP] {k[:62]!r:<66} x{len(v)} @{v[0]}")
        for k in sorted(cov):
            print(f"   ok    {k[:62]!r}")
        return {"covered": cov, "gap": gap}

    res = {"dialog_window_title": dump("dialog:{window:{title}}", titles),
           "dialog_buttons_label": dump("dialog:{buttons:{…label}}", labels),
           "configureDialog_dynamic": dump("_configureDialog 内动态 label/title", dyn)}
    json.dump(res, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
