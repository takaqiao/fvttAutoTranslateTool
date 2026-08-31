# -*- coding: utf-8 -*-
r"""
probe: i18n_autoloc_sink_gap2 —— 上一支的**收窄版**（只读）

sink_gap（第一支）是「属性名匹配」，噪声大：ember 里 5003 个 `label:` 绝大多数
是音景/剧情**数据**，根本不过 localize。

这一支改成**调用点定域**：只在 core 会**无条件**调 `_loc()` 的那几个声明式面里取值。
每个面都附 core 源码证据行（v14，`C:\Program Files\Foundry Virtual Tabletop\resources\app`）：

  D1  DialogV2 按钮 label / tooltip
      client/applications/api/dialog.mjs:249  `span.innerText = _loc(label);`
      client/applications/api/dialog.mjs:240  `button.setAttribute("aria-label", _loc(tooltip));`
  D2  DialogV2 / ApplicationV2 窗口标题
      client/applications/api/application.mjs:320  `get title() { return _loc(this.options.window.title); }`
  D3  ApplicationV2 窗口头部控件 label
      client/applications/api/application.mjs:910  `span.innerText = _loc(control.label);`
  D4  游戏设置 name / hint
      client/applications/settings/config.mjs:126-127
      `data.field.label ||= _loc(setting.name ?? ""); data.field.hint ||= _loc(setting.hint ?? "");`
  D5  ContextMenu 条目 label / name
      client/applications/ux/context-menu.mjs:403  `const name = _loc("label" in item ? item.label : item.name);`
  D6  data-tooltip（**条件**本地化，但同属本类：加了键就会被翻）
      client/helpers/interaction/tooltip-manager.mjs:261-263
      `text = element.dataset.tooltip || element.ariaLabel; if ( game.i18n.has(text) ) ... _loc(text)`

取值后同样做三步差集：不在 core/crucible/ember 三张 en.json 拍平键集里、不形如 A.B.C、
本项目 cn.json / 硬编码表也没兜住 → 界面永远英文，且现有判据全都看不见。

假阳性模式：
  · 定域用的是「调用点后 N 个字符」的窗口，可能吃进相邻代码里的同名属性（已打印上下文供核对）；
  · `label:` 有可能是变量而非字面量（本脚本只抓字面量，属漏报不属误报）；
  · 死代码 / 仅开发者可达。

只读，不写库。
"""
import io
import json
import os
import re
import sys

CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_autoloc_sink_gap2.json")

JS = {
    "crucible": os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs"),
    "ember": os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs"),
}
TPL = {
    "crucible": os.path.join(FVTT, "systems", "crucible", "templates"),
    "ember": os.path.join(FVTT, "modules", "ember", "templates"),
}
EN_JSONS = [os.path.join(CORE, "public", "lang", "en.json"),
            os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
            os.path.join(FVTT, "modules", "ember", "lang", "en.json")]
CN_JSONS = [os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
            os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json")]
CN_JS = [os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
         os.path.join(ROOT, "1-Ember汉化插件", "register.js"),
         os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js")]

# 定域锚点：正则 -> 从匹配处向后取多少字符
ANCHORS = [
    ("D1/D2 DialogV2", re.compile(r"DialogV2[\w$]*\s*\.\s*(?:prompt|confirm|wait|query)\s*\(|new\s+DialogV2[\w$]*\s*\("), 2200),
    ("D1/D2 DialogV2", re.compile(r"foundry\.applications\.api\.DialogV2\b"), 2200),
    ("D3 headerControls", re.compile(r"_getHeaderControls\s*\(\s*\)\s*\{|controls\s*:\s*\["), 1400),
    ("D4 settings", re.compile(r"game\.settings\.register(?:Menu)?\s*\(|game\.keybindings\.register\s*\("), 900),
    ("D5 contextMenu", re.compile(r"ContextMenu|_getEntryContextOptions|ContextMenuEntry|getContextMenu"), 1600),
]
PROP = re.compile(r"(?<![\w$.])(label|title|tooltip|name|hint)\s*:\s*([\"'])((?:[^\"'\\\n]|\\.)*)\2")
KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$\-]+)+$")
TIP_ATTR = re.compile(r'data-tooltip\s*=\s*"([^"{}]+)"')


def flat(o, p=""):
    s = set()
    if isinstance(o, dict):
        for k, v in o.items():
            q = f"{p}.{k}" if p else k
            s.add(q)
            s |= flat(v, q)
    return s


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def main():
    en = set()
    for p in EN_JSONS:
        en |= flat(json.load(io.open(p, encoding="utf-8")))
    cn = set()
    for p in CN_JSONS:
        cn |= flat(json.load(io.open(p, encoding="utf-8")))
    cnlit = set()
    for p in CN_JS:
        if os.path.exists(p):
            s = read(p)
            cnlit |= set(re.findall(r"[\"'`]((?:[^\"'`\\]|\\.){2,120}?)[\"'`]\s*:", s))
            for m in re.finditer(r"game\.i18n\.translations(?:\.([\w$]+)|\[[\"']([^\"']+)[\"']\])", s):
                cn.add(m.group(1) or m.group(2))

    def gap(t):
        t = t.strip()
        if not t or len(t) > 90:
            return False
        if t in en or t in cn or t in cnlit:
            return False
        if KEYISH.match(t) or "${" in t or "{{" in t:
            return False
        if not re.search(r"[A-Za-z]{2,}", t):
            return False
        return bool(re.search(r"[A-Z]", t) or " " in t)

    result = {}
    for who, path in JS.items():
        s = read(path)
        seen = {}
        for face, rx, span in ANCHORS:
            for m in rx.finditer(s):
                a, b = m.start(), min(len(s), m.end() + span)
                for pm in PROP.finditer(s, a, b):
                    t = pm.group(3)
                    if not gap(t):
                        continue
                    ln = s[:pm.start()].count("\n") + 1
                    key = (face, pm.group(1), t, ln)
                    if key in seen:
                        continue
                    seen[key] = {"face": face, "prop": pm.group(1), "text": t, "line": ln,
                                 "ctx": s[max(0, pm.start() - 200):pm.end() + 120].replace("\n", " ⏎ ")}
        # D6 模板里的 data-tooltip 常量
        tips = []
        for base, _dn, fns in os.walk(TPL[who]):
            for fn in fns:
                if not fn.endswith((".hbs", ".html")):
                    continue
                fp = os.path.join(base, fn)
                t = read(fp)
                for m in TIP_ATTR.finditer(t):
                    if gap(m.group(1)):
                        tips.append({"face": "D6 data-tooltip(tpl)", "prop": "data-tooltip",
                                     "text": m.group(1), "line": t[:m.start()].count("\n") + 1,
                                     "ctx": os.path.relpath(fp, FVTT)})
        rows = sorted(seen.values(), key=lambda r: r["line"]) + tips
        result[who] = rows
        print("=" * 96)
        print(f"[{who}] 定域命中的无键裸英文 = {len(rows)}")
        for r in rows:
            print(f"  {r['face']:<22} {r['prop']:<9} L{r['line']:<7} {r['text']!r}")
            print(f"      {r['ctx'][:250]}")
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
